import torch
from torch import nn

def gcu(input):
    '''
    Applies the Growing Cosine Unit (GCU) function element-wise:
        SiLU(x) = x * cos(x)
    '''

    return input * torch.cos(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

class GCU(nn.Module):
    '''
    Applies the Growing Cosine Unit (GCU) function element-wise:
        SiLU(x) = x * cos(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = gcu()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return gcu(input) # simply apply already implemented SiLU
