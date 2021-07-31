import torch
from torch.optim.optimizer import Optimizer

from utils.kfrac_utils import (ComputeCovA, ComputeCovG)
from utils.kfac_utils import update_running_stat

class SAM_KFAC(Optimizer):
    r"""Implements stochastic gradient descent (optionally with Sharpness-aware Minimization).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 0.1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum (float, optional): momentum factor (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nbs (float): Neighborhood size (default: 0)

    Example:
        >>> optimizer = sam.SAM(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9, nbs=0.1)
        >>> optimizer.zero_grad()
        >>> loss = loss_fn(model(input), target)
        >>> original_params = network.parameters()
        >>> loss.backward()
        >>> optimizer.step_calc_w_adv()
        >>> loss_sam = loss_fn(model(input), target)
        >>> loss_sam.backward()
        >>> optimizer.load_original_params_and_step(original_params)
    """

    def __init__(self,
                 model,
                 lr=0.1,
                 momentum=0,
                 nesterov=False,
                 nbs=0,
                 start_decay=0.95
                 dampening=0,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if nbs < 0.0:
            raise ValueError("Invalid neighborhood_size value: {}".format(nbs))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        nesterov=nesterov, weight_decay=weight_decay, nbs=nbs)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SAM_KFAC, self).__init__(model.parameters(), defaults)

        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.know_modulers = {"Linear", "Conv2d"}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv

    ##########################################################
    # These are methods used in KFAC                         #
    ##########################################################
    
    def _save_input(self,module,input):
        """
        This method calculates the CovA matrix for the specified module and using the specified input

        m_aa dictionary would be updated 
        update_running_stat is called 
        """
        
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        """
        This method calculates the CovG matrix for the specified module and using the in_grad and out_grad 

        m_gg dictionary will be updated 
        update_running_stat is called 
        """
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        """
        It will fill the self.modules array with the modules of the model ( just the Linear and Conv2d modules) as well as adding the hooks (_save_input and _save_grad_output) to each module. 
        """
        
        count = 0
        print(self.model)
        print("=> We keep following layers in KFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in KFAC. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        eps = 1e-10  # for numerical stability
        self.d_a[m], self.Q_a[m] = torch.symeig(
            self.m_aa[m], eigenvectors=True)
        self.d_g[m], self.Q_g[m] = torch.symeig(
            self.m_gg[m], eigenvectors=True)

        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']


                        buf.mul_(momentum).add_(1, d_p)
                    d_p = buf

                p.data.add_(-group['lr'], d_p)

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                self._update_inv(m)
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1
    
    ##########################################################
    # These are methods used in SAM                          #
    ##########################################################
    
    def __setstate__(self, state):
        super(SAM_KFAC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step_calc_w_adv(self, closure=None):
        """Update models weight params with w_adv.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            nbs = group['nbs']
            grad_l2norm = 0

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                grad_l2norm += torch.norm(d_p)**2
            if grad_l2norm == float('inf'):
                print('!!!grad_l2norm is too large!!!')
            if torch.isnan(grad_l2norm):
                print('!!!grad_l2norm is nan!!!')
            grad_l2norm = grad_l2norm**(1/2)

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                p.data = (p.data + (nbs/grad_l2norm)*d_p).clone().detach().requires_grad_(True)

        return loss

    @torch.no_grad()
    def load_original_params_and_step(self, original_params, closure=None):
        """Load original params (w_t) without gradient information and update weights params with SAM gradient

        Arguments:
            original_params: The original model params
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for i,p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.data = (original_params[i].data - lr*d_p).clone().detach().requires_grad_(True)

        return loss

