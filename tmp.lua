require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'module/normalConv'
require 'module/normalLinear'
require 'sys'
dofile 'etc.lua'
dofile 'data.lua'

a = torch.Tensor({1,2,3,4,5})
b = torch.Tensor({11,12,13,14,15})

c = torch.cmul(a:gt(0),b:lt(13)):eq(0)
print(c)

