require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'module/normalConv'
require 'module/normalLinear'
require 'sys'
dofile 'etc.lua'


--[===[
a = torch.CudaTensor({8,5,7,9})
--a:zero()
print(a)
--print(a:topk(5,true))
print(a[{{1,3}}])
--b = torch.Tensor({{{1,2},{3,4}},{{5,6},{7,8}}})
--print(b)
--print(torch.reshape(b,8))

c = torch.range(1,10)
print(c)
print(c:size())
print(c:size()[1])
--]===]
--
--
input = torch.CudaTensor(2,20,7,7):fill(0.5)
print(cudnn.SpatialLogSoftMax:cuda():forward(input))
