require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'module/normalConv'
require 'module/normalLinear'
dofile 'etc.lua'


----------------------------
VGG_pretrain = torch.load("/home/mks0601/workspace/Model/vgg_pretrain.t7")

VGGNet = nn.Sequential()

for i = 1,31 do

    if i == 5 or i == 10  or i == 24  then
        VGGNet:add(nn.SpatialMaxPooling(2,2,2,2))
    elseif i == 17 then
        VGGNet:add(nn.SpatialMaxPooling(2,2,2,2,1,1))
    elseif i == 31 then
        VGGNet:add(nn.SpatialMaxPooling(3,3,1,1,1,1))
    else
        VGGNet:add(VGG_pretrain:get(i))
    end

end


------------------------------
mainBranch_1 = nn.Sequential()

kernelSz = 3
prev_fDim = 512
next_fDim = 1024
mainBranch_1:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
mainBranch_1:add(nn.ReLU(true))
kernelSz = 1
mainBranch_1:add(cudnn.normalConv(next_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*next_fDim))))
mainBranch_1:add(nn.ReLU(true))

subBranch_2 = nn.Sequential()
kernelSz = 3
prev_fDim = 1024
next_fDim = 6*(classNum+4)
subBranch_2:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
----------------------------
mainBranch_2 = nn.Sequential()
kernelSz = 1
prev_fDim = 1024
next_fDim = 256
mainBranch_2:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
mainBranch_2:add(nn.ReLU(true))

kernelSz = 3
prev_fDim = 256
next_fDim = 512
mainBranch_2:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,2,2,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
mainBranch_2:add(nn.ReLU(true))

subBranch_3 = nn.Sequential()
kernelSz = 3
prev_fDim = 512
next_fDim = 6*(classNum+4)
subBranch_3:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
--------------------------
mainBranch_3 = nn.Sequential()
kernelSz = 1
prev_fDim = 512
next_fDim = 128
mainBranch_3:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
mainBranch_3:add(nn.ReLU(true))
kernelSz = 3
prev_fDim = 128
next_fDim = 256
mainBranch_3:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,2,2,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
mainBranch_3:add(nn.ReLU(true))

subBranch_4 = nn.Sequential()
kernelSz = 3
prev_fDim = 256
next_fDim = 6*(classNum+4)
subBranch_4:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
-------------------------------
mainBranch_4 = nn.Sequential()
kernelSz = 1
prev_fDim = 256
next_fDim = 128
mainBranch_4:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
mainBranch_4:add(nn.ReLU(true))
kernelSz = 3
prev_fDim = 128
next_fDim = 256
mainBranch_4:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,2,2,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
mainBranch_4:add(nn.ReLU(true))

subBranch_5 = nn.Sequential()
kernelSz = 3
prev_fDim = 256
next_fDim = 6*(classNum+4)
subBranch_5:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
-------------------------
mainBranch_5 = nn.Sequential()
mainBranch_5:add(nn.SpatialAveragePooling(3,3))
mainBranch_5:add(nn.ReLU(true))

subBranch_6 = nn.Sequential()
kernelSz = 1
prev_fDim = 256
next_fDim = 5*(classNum+4)
subBranch_6:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
--------------------------------
model = nn.Sequential():add(VGGNet):add(mainBranch_1)

concat = nn.ConcatTable()
concat:add(subBranch_2)
concat:add(mainBranch_2)
model:add(concat)

concat = nn.ConcatTable()
concat:add(nn.SelectTable(1))
concat:add(nn.Sequential():add(nn.SelectTable(2)):add(subBranch_3))
concat:add(nn.Sequential():add(nn.SelectTable(2)):add(mainBranch_3))
model:add(concat)

concat = nn.ConcatTable()
concat:add(nn.SelectTable(1))
concat:add(nn.SelectTable(2))
concat:add(nn.Sequential():add(nn.SelectTable(3)):add(subBranch_4))
concat:add(nn.Sequential():add(nn.SelectTable(3)):add(mainBranch_4))
model:add(concat)

concat = nn.ConcatTable()
concat:add(nn.SelectTable(1))
concat:add(nn.SelectTable(2))
concat:add(nn.SelectTable(3))
concat:add(nn.Sequential():add(nn.SelectTable(4)):add(subBranch_5))
concat:add(nn.Sequential():add(nn.SelectTable(4)):add(mainBranch_5))
model:add(concat)

concat = nn.ConcatTable()
concat:add(nn.SelectTable(1))
concat:add(nn.SelectTable(2))
concat:add(nn.SelectTable(3))
concat:add(nn.SelectTable(4))
concat:add(nn.Sequential():add(nn.SelectTable(5)):add(subBranch_6))
model:add(concat)

---------------------------
crossEntropy = nn.CrossEntropyCriterion()
smoothL1 = nn.SmoothL1Criterion()
smoothL1.sizeAverage = false

cudnn.convert(model, cudnn)
model:cuda()
crossEntropy:cuda()
smoothL1:cuda()
cudnn.fastest = true
cudnn.benchmark = true
