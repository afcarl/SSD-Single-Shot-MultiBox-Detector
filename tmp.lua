require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'module/normalConv'
require 'module/normalLinear'
require 'sys'
dofile 'etc.lua'

w = 2
h = 2
dim = 3
input = torch.reshape(torch.range(1,36),3,dim,h,w)
print(input)

model = nn.Sequential()

transpose = nn.Transpose({2,3},{3,4})
reshape = nn.Reshape(w*h*dim)
reshape2 = nn.Reshape(w*h,dim,1)

splittable = nn.SplitTable(1,3)

normalize = nn.Normalize(2)
map = nn.MapTable()
map:add(nn.Sequential():add(nn.Reshape(dim)):add(normalize):add(nn.Reshape(dim,1,1,true)))

jointable = nn.JoinTable(1,3)

model:add(transpose):add(reshape):add(reshape2)
model:add(splittable)
model:add(map)
model:add(jointable)
model:add(nn.Reshape(w*h,dim))
model:add(nn.Reshape(w,h,dim))
model:add(nn.Transpose({3,4},{2,3}))


result = model:forward(input)
print(result)



