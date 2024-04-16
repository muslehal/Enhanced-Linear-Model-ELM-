import torch
from torch import nn

class RevIN(nn.Module):
    #the initializer method, which accepts three parameters: the number of features or channels, eps for numerical stability,
    # and affine which is a boolean indicating whether the layer has learnable affine parameters.
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:#If the affine flag is true, the _init_params method is called to initialize the parameters for the affine transformation.
            self._init_params()

    def forward(self, x, mode:str):  #The mode parameter determines whether normalization or denormalization is performed.
        if mode == 'norm':
            self._get_statistics(x)#If the mode is 'norm', the _get_statistics method is called to calculate the mean and standard deviation of x.
            x = self._normalize(x)
        elif mode == 'denorm':   #If the mode is 'denorm', the _denormalize method is called to perform the reverse operation of normalization.
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self): 
        # initialize RevIN params: (C,)
        #This method initializes the parameters for the affine transformation, which are a weight and bias vector of size equal to the number of channels.
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        #torch.Size([42, 16, 147])
        #This method computes the mean and standard deviation of x over all dimensions except the batch and channel dimensions.
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x): 
        #This method normalizes x by subtracting the mean and dividing by the standard deviation.
        # If the affine flag is true, an affine transformation is applied.
        # x torch.Size([64, 16, 147])
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        #This method performs the reverse operation of normalization. If the affine flag is true, it first reverses the affine transformation,
        # then multiplies by the standard deviation and adds the mean.
        #torch.Size([42, 96, 7])
        # xtorch.Size([64, 16, 147])
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

