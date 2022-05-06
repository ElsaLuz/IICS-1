import torch
import torch.nn as nn
from torch.nn import init # To initialize the weights of a single layer, use a function from torch.nn.init
from torchvision import models
from .backbones.resnet import AIBNResNet


def weights_init_kaiming(m): #Kaiming works very well with Relu. https://www.youtube.com/watch?v=tMjdQLylyGI #https://www.youtube.com/watch?v=xWQ-p_o0Uik
    classname = m.__class__.__name__ #https://www.tutorialspoint.com/What-does-built-in-class-attribute-name-do-in-Python   
                                     # m: instance, __class__: We use the __class__ property of the object to find the type or class of the object.
                                     #__class__ is an attribute on the object that refers to the class from which the object was created.
                                     # __name__ : https://www.youtube.com/watch?v=pzNISmtmzcY
                                     # https://teamtreehouse.com/community/classname-2
    # print(classname)
    if classname.find('Conv') != -1:    #The find() method returns -1 if the value is not found.
        init.kaiming_normal_(
            m.weight.data, a=0, # m.weight.data means changing weights of this layer  # a – the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
            mode='fan_in')  # For old pytorch, you may use kaiming_normal. # either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0) # Fills the input Tensor with the value val .
    elif classname.find('BatchNorm1d') != -1: # 2d for image , 1D for flatenned layers
        init.normal_(m.weight.data, 1.0, 0.02) # tensor, mean, std
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001) # mean=0
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


class ft_net_intra(nn.Module):
    def __init__(self, num_classes, stride=1):
        super(ft_net_intra, self).__init__() #if you want the __init__ to run of your parent class that you are inheriting from, you run super().__init()
        model_ft = AIBNResNet(last_stride=stride,
                              layers=[3, 4, 6, 3])

        self.model = model_ft
        self.classifier = nn.ModuleList( # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/6 #https://www.coursera.org/lecture/deep-neural-networks-with-pytorch/8-1-2-deeper-neural-networks-nn-modulelist-ECAbQ
            [nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, num, bias=False))
             for num in num_classes])
        for classifier_one in self.classifier:
            init.normal_(classifier_one[1].weight.data, std=0.001)
            init.constant_(classifier_one[0].weight.data, 1.0)
            init.constant_(classifier_one[0].bias.data, 0.0)
            classifier_one[0].bias.requires_grad_(False)

    def backbone_forward(self, x):
        x = self.model(x)
        return x

    def forward(self, x, k=0):
        x = self.backbone_forward(x)
        x = x.view(x.size(0), x.size(1)) # reshaping (aruments: rows, columns) #For example, if the shape of x were (10,20) then x.size(1) refers to second dimension i.e. 20. 
        x = self.classifier[k](x)
        return x


class ft_net_inter(nn.Module):
    def __init__(self, num_classes, stride=1):
        super(ft_net_inter, self).__init__()
        model_ft = AIBNResNet(last_stride=stride,
                              layers=[3, 4, 6, 3])

        self.model = model_ft # until here we've got the features
        self.classifier = nn.Sequential(nn.BatchNorm1d( 
            2048), nn.Linear(2048, num_classes, bias=False)) # (512*4)
        init.normal_(self.classifier[1].weight.data, std=0.001) 
        init.constant_(self.classifier[0].weight.data, 1.0)
        init.constant_(self.classifier[0].bias.data, 0.0)
        self.classifier[0].bias.requires_grad_(False)

    def backbone_forward(self, x):
        x = self.model(x)
        return x

    def forward(self, x):
        x = self.backbone_forward(x)
        x = x.view(x.size(0), x.size(1)) #flattening the tensor to be connected with fc, https://ofstack.com/python/40552/interpretation-of-x-=-x.-view-of-x.-size-of-0--1-in-pytorch.html
        prob = self.classifier(x)
        return prob, x
