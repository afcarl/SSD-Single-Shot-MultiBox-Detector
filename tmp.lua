require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'module/normalConv'
require 'module/normalLinear'
require 'sys'
dofile 'etc.lua'


a = torch.Tensor({1,2,3})
b = torch.Tensor({4,5,6})
print(torch.gt(a,b))
