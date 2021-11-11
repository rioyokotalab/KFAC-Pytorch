import torchvision

from models.cifar import (alexnet, densenet, resnet,
                          vgg16_bn, vgg19_bn,
                          wrn)


def get_network(network, **kwargs):
    networks = {
        'alexnet': alexnet,
        'densenet': densenet,
        'resnet': resnet,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'wrn': wrn

    }
    if network == 'resnet50':
        return torchvision.models.resnet50()
    return networks[network](**kwargs)

