import torch
import torch.nn as nn

# The self parameter is a reference to the current instance of the class, and is used to access variables that belongs to the class.
#It does not have to be named self , you can call it whatever you like, but it has to be the first parameter of any function in the class
#https://www.youtube.com/watch?v=M1BAlDufqao
class AIBNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, only_bn=False,
                 last_gamma=False, adaptive_weight=None, generate_weight=False):
        super(AIBNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.only_bn = only_bn
        self.last_gamma = last_gamma
        self.generate_weight = generate_weight
        if generate_weight: # if generate_weight has a value then execute the statements underneath
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        if not only_bn:
            if adaptive_weight is not None:
                self.adaptive_weight = adaptive_weight
            else:
                self.adaptive_weight = nn.Parameter(torch.ones(1) * 0.1)
        #a buffer can be found in _BatchNorm module where the running_mean , running_var and num_batches_tracked are 
        #registered as buffers and updated by accumulating statistics of data forwarded through the layer.
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.reset_parameters()

    def reset_parameters(self):

        self.running_mean.zero_() #zeros out the mean
        self.running_var.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x, weight=None, bias=None):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1) # (N, C, H*W)
        # In “Instance Normalization”, mean and variance are calculated for each individual channel for each individual sample across both spatial dimensions
        mean_in = x.mean(-1, keepdim=True) # calculate means across last dimension, keepdim = output has the dimension retained in this case (True)
        var_in = x.var(-1, keepdim=True)

        temp = var_in + mean_in ** 2

        if self.training:
            mean_bn = mean_in.mean(0, keepdim=True)
            var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_(
                    (1 - self.momentum) * mean_bn.squeeze().data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum)
                                      * var_bn.squeeze().data) # removing all of the axes having length of 1
            else:
                self.running_mean.add_(mean_bn.squeeze().data)
                self.running_var.add_(
                    mean_bn.squeeze().data ** 2 + var_bn.squeeze().data)
        else: # 
            mean_bn = torch.autograd.Variable(
                self.running_mean).unsqueeze(0).unsqueeze(2)
            var_bn = torch.autograd.Variable(
                self.running_var).unsqueeze(0).unsqueeze(2)

        if not self.only_bn:

            adaptive_weight = torch.clamp(self.adaptive_weight, 0, 1) # clamping alpha during forward pass to avoid negative values
            mean = (1 - adaptive_weight[0]) * \
                mean_in + adaptive_weight[0] * mean_bn
            var = (1 - adaptive_weight[0]) * \
                var_in + adaptive_weight[0] * var_bn

            x = (x-mean) / (var + self.eps).sqrt()  # eq 12
            x = x.view(N, C, H, W)
        else:       #may be back propagation
            x = (x - mean_bn) / (var_bn + self.eps).sqrt()
            x = x.view(N, C, H, W)

        if self.generate_weight:
            weight = self.weight.view(1, self.num_features, 1, 1)
            bias = self.bias.view(1, self.num_features, 1, 1)
        else:
            weight = weight.view(1, self.num_features, 1, 1)
            bias = bias.view(1, self.num_features, 1, 1)
        return x * weight + bias
