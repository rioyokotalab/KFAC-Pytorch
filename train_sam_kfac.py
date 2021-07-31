import argparse
import os
import sys
import random
import time
import warnings
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import model_loader
import math
import sam

import wandb

from myutils import SmallIterator

from opitmizers import (KFACOptimizer, EKFACOptimizer)

parser = argparse.ArgumentParser(description='PyTorch test distributed SAM Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('--nbs', '--neighborhood-size', default=0.05, type=float,
                    metavar='S', help='neighborhood size', dest='nbs')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 30)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--experiment',type=str,default='experiment',help="Name of the experiment")

best_acc = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('slow work because of to apply seed setting')

    args.distributed = True
    main_worker(args)


def main_worker(args):
    global best_acc

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    # if args.distributed:
    #     # initialize torch.distributed using MPI
    #     from mpi4py import MPI
    #     comm = MPI.COMM_WORLD
    #     world_size = comm.Get_size()
    #     rank = comm.Get_rank()
    #     ngpus_per_node = torch.cuda.device_count()
    #     device = rank % ngpus_per_node
    #     print(f'rank : {rank}    world_size : {world_size}')
    #     torch.cuda.set_device(device)
    #     init_method = 'tcp://{}:23456'.format(args.dist_url)
    #     torch.distributed.init_process_group('nccl', init_method=init_method, world_size=world_size, rank=rank)
    #     args.rank = rank
    #     args.gpu = device
    #     args.world_size = world_size

    if args.distributed:
        master_addr = os.getenv("MASTER_ADDR", default="localhost")
        master_port = os.getenv('MASTER_PORT', default='8888')
        method = "tcp://{}:{}".format(master_addr, master_port)
        rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
        world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))
        ngpus_per_node = torch.cuda.device_count()
        device = rank % ngpus_per_node
        print(f'rank : {rank}    world_size : {world_size}')
        torch.cuda.set_device(device)
        torch.distributed.init_process_group('nccl', init_method=method, world_size=world_size, rank=rank)
        args.rank = rank
        args.gpu = device
        args.world_size = world_size

    # LR=int(args.batch_size/64)
    # # le=(len(bin(LR))-bin(LR).rfind("1")-1)
    # # args.lr=args.lr*2**(le/2)
    # args.lr=args.lr*LR

    config = {}
    if args.rank == 0:
        # Init wandb
        wandb.init(
            project="SAM_KFAC_tests",
            config={
                "global_batch_size": args.batch_size,
                "local_batch_size": int(args.batch_size / args.world_size),
                "np": args.world_size,
                "weight_decay": args.weight_decay,
                "initial_learn_rate": args.lr,
                "momentum": args.momentum,
                "nbs": args.nbs
            },
            name=args.experiment
        )
        config = wandb.config

    # create model
    arch = "WRN-28-10"
    print("=> creating model '{}'".format(arch))
    model = model_loader.WRN(28, 10, 0, 10)
    # init model layers
    model.apply(model_loader.conv_init)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + args.world_size - 1) / args.world_size)
            process_group = torch.distributed.new_group([i for i in range(args.world_size)])
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = sam.SAM(model.parameters(),
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        nbs=args.nbs)

    

    cudnn.benchmark = True


    # '''test'''
    # def nan_hook(self, inp, output):
    #     if not isinstance(output, tuple):
    #         outputs = [output]
    #     else:
    #         outputs = output
    #
    #     for i, out in enumerate(outputs):
    #         nan_mask = torch.isnan(out)
    #         if nan_mask.any():
    #             print("In", self.__class__.__name__)
    #             raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:",
    #                                out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
    # for submodule in model.modules():
    #     submodule.register_forward_hook(nan_hook)
    # '''test'''


    # Data loading code
    traindir = './cifar10_data/train'
    valdir = './cifar10_data/validation'

    # データ正規化
    train_form = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.4914, 0.4822, 0.4465],  # RGB 平均
            [0.2023, 0.1994, 0.2010]  # RGB 標準偏差
        )
    ])

    test_form = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.4914, 0.4822, 0.4465],  # RGB 平均
            [0.2023, 0.1994, 0.2010]  # RGB 標準偏差
        )
    ])

    # load CIFAR10 data
    train_dataset = datasets.CIFAR10(  # CIFAR10 default dataset
        root=traindir,  # rootで指定したフォルダーを作成して生データを展開。これは必須。
        train=True,  # 学習かテストかの選択。これは学習用のデータセット
        transform=train_form,
        download=True
    )

    test_dataset = datasets.CIFAR10(
        root=valdir,
        train=False,  # 学習かテストかの選択。これはテスト用のデータセット
        transform=test_form,
        download=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=args.world_size,
                                                                        rank=args.rank)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                       num_replicas=args.world_size,
                                                                       rank=args.rank)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler)

    if args.rank == 0:
        print(args)

    if args.evaluate:
        validate(test_loader, model, criterion, args)
        return

    def learning_rate(init, epoch):
        optim_factor = 0
        if (epoch > 160):
            optim_factor = 3
        elif (epoch > 120):
            optim_factor = 2
        elif (epoch > 60):
            optim_factor = 1

        return init * math.pow(0.2, optim_factor)

    if args.rank == 0:
        # make models save dir if not exist
        os.makedirs(f'./trained_models/WRN-28-10_with_original_SAM', exist_ok=True)

        # prepare for saving the model
        save_list = []
        save_list.append(f'epochs={args.epochs}')
        save_list.append(f'lr={args.lr}')
        save_list.append(f'momentum={args.momentum}')
        save_list.append(f'weight_decay={args.weight_decay}')
        save_list.append(f'nbs={args.nbs}')
        save_list.append(f'global_batch={config["global_batch_size"]}')
        save_list.append(f'local_batch={args.batch_size}')
        model_path = ''.join(
            ['./trained_models/WRN-28-10_with_original_SAM/model_', '_'.join(save_list), '.ckpt'])
        optim_path = ''.join(
            ['./trained_models/WRN-28-10_with_original_SAM/optim_', '_'.join(save_list), '.ckpt'])

    if args.rank == 0:
        wandb.log({"model_path": model_path, "optim_path": optim_path})

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate(args.lr, epoch+1)

        # train for one epoch
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        test_acc, test_loss = validate(test_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if is_best and args.rank == 0:
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optim_path)

        if args.rank == 0:
            wandb.log({"train_acc": train_acc,
                       "train_loss": train_loss,
                       "test_acc": test_acc,
                       "test_loss": test_loss,
                       "epoch_lr": learning_rate(args.lr, epoch+1)
                       })

    if args.rank == 0:
        print('---------DONE---------')
        print('FINAL BEST test_accuracy: {} %'.format(best_acc))
        wandb.log({"best_test_acc": best_acc})

        # save parameters, model_path
        log_dict = {"model_path": model_path,
                    "optim_path": optim_path}
        print(log_dict)
        print(args)


# def save_model_buffers(model):
#     return [buf.data.clone().detach() for buf in model.buffers()]


# def load_model_buffers(model, buf_list):
#     for i,buf in enumerate(model.buffers()):
#         buf.data = buf_list[i]

def save_model_buffers(model):
    ret = {}
    for k,v in model.state_dict().items():
        if 'mean' in k or 'var' in k or 'num_batches_tracked' in k:
            ret["model." + k + ".data"] = v.data.clone()
    return ret

def load_model_buffers(model, buf_dict):
    for k,v in buf_dict.items():
      exec("%s = v" % (k))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # save w_t params
        w_t = [param.clone().detach().requires_grad_(True) for param in model.parameters()]

        # reset gradient
        optimizer.zero_grad()

        # compute output and normal_grad each sam_local_batch without all-reduce normal_grad
        # normal_gradはsam_bnごとに計算し，それぞれw_advを出すのに使うので，この時の同期は切断(all-reduceしない)
        # この時のbatch normの平均と分散は．globalにsync (Sync Batch Normを使用している)
        # ここでのbatch normの平均と分散を更新に反映
        # ここのforward計算前にはw_tは揃っているはず．(normal gradを用いたstep更新では各プロセスは全く同じ計算をしているはず．)

        small_ds = SmallIterator(images,target,256)
        loss = torch.tensor(0.0)
        output = torch.tensor([])
        num_small_batch = len(small_ds)
        ret = None
        if args.gpu is not None:
            loss = loss.cuda(args.gpu)
            output = output.cuda(args.gpu)
        with model.no_sync():
            for (small_images,small_target) in small_ds:
                small_output = model(small_images)
                output = torch.cat((output,small_output))
                # calc normal_grad
                small_loss = criterion(small_output, small_target)
                small_loss /= num_small_batch
                small_loss.backward()
                if ret is None:
                    ret = save_model_buffers(model)
                    for k in ret:
                        if not "num_batches" in k:
                            ret[k] /= num_small_batch
                else:
                    tmp_ret = save_model_buffers(model)
                    for k in tmp_ret:
                        if not "num_batches" in k:
                            ret[k] += tmp_ret[k]/num_small_batch
                loss += small_loss
            
            load_model_buffers(model,ret)
            # compute w_adv from normal_grad
            optimizer.step_calc_w_adv()

        acc1 = accuracy(output, target)[0]
        # save model buffers (for Sync Batch Norm info)
        buf_list = save_model_buffers(model)

        # reset gradient
        optimizer.zero_grad()

        last_small_idx = num_small_batch - 1
        ret = None
        for small_idx,(small_images, small_target) in enumerate(small_ds):
            # この時のbatch normの平均と分散は．globalにsync (Sync Batch Normを使用している)
            # ここでのbatch normの平均と分散の更新は反映しない．(後に430行目でresetする．)
            if small_idx != last_small_idx:
                with model.no_sync():
                    small_output = model(small_images)

                    # calc sam loss (each sam batch)
                    loss_sam = criterion(small_output, small_target)

                    loss_sam /= num_small_batch
                    
                    # calc global sam_grad (with all-reduce sam_grad each sam batch)
                    loss_sam.backward()
            else:
                small_output = model(small_images)
                
                # calc sam loss (each sam batch)
                loss_sam = criterion(small_output, small_target)

                loss_sam /= num_small_batch
                
                # calc global sam_grad (with all-reduce sam_grad each sam batch)
                loss_sam.backward()
                

        # load w_t weights params (without gradient information)
        # update weights params from sam grad
        # 元の位置(w_t)からglobal sam_gradを用いて更新！
        optimizer.load_original_params_and_step(w_t)

        # この処理でacc1は平均されて処理されるはず！
        # この処理でlossも平均されて処理されるはず！
        if args.distributed:
            size = float(args.world_size)
            dist.all_reduce(acc1, op=dist.ReduceOp.SUM)
            acc1 /= size
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= size

        # update loss and acc info
        top1.update(acc1[0], images.size(0))
        losses.update(loss.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0:
            if i % args.print_freq == 0:
                progress.display(i)

        # load model buffers (reset BN info because computed BN info again)
        load_model_buffers(model, buf_list)

    return top1.avg, losses.avg


def validate(test_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            small_ds = SmallIterator(images,target,256)
            num_small_batch = len(small_ds)
            
            output = torch.tensor([])
            loss = torch.tensor(0.0)
            if args.gpu is not None:
                output = output.cuda(args.gpu)
                loss = loss.cuda(args.gpu)
            
            for small_images,small_target in small_ds:
                # compute output
                small_output = model(small_images)
                output = torch.cat((output,small_output))
                small_loss = criterion(small_output, small_target)
                small_loss /= num_small_batch
                loss += small_loss
                # measure accuracy and record loss

            acc1 = accuracy(output, target)[0]

            # この処理でacc1は平均されて処理されるはず！
            # この処理でlossも平均されて処理されるはず！
            if args.distributed:
                size = float(args.world_size)
                dist.all_reduce(acc1, op=dist.ReduceOp.SUM)
                acc1 /= size
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= size

            # update loss and acc info
            top1.update(acc1[0], images.size(0))
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.rank == 0:
                if i % args.print_freq == 0:
                    progress.display(i)

        if args.rank == 0:
            # TODO: this should also be done with the ProgressMeter
            print(' * Test Acc (avg.) {top1.avg:.3f}'
                  .format(top1=top1))

    return top1.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    p = multiprocessing.Process()
    p.start()
    main()
