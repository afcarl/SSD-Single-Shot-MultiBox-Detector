require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'loadcaffe'
require 'module/normalConv'
require 'module/normalDilatedConv'
require 'module/normalLinear'
require 'module/customCMul'
dofile 'etc.lua'


function build_l2_normalize(dim)
    
    model = nn.Sequential()

    transpose = nn.Transpose({2,3},{3,4})
    reshape = nn.Reshape(fmSz[1]*fmSz[1]*dim)
    reshape2 = nn.Reshape(fmSz[1]*fmSz[1],dim,1)

    splittable = nn.SplitTable(1,3)

    normalize = nn.Normalize(2)
    map = nn.MapTable()
    map:add(nn.Sequential():add(nn.Reshape(dim)):add(normalize):add(nn.Reshape(dim,1,1,true)))

    jointable = nn.JoinTable(1,3)

    model:add(transpose):add(reshape):add(reshape2)
    model:add(splittable)
    model:add(map)
    model:add(jointable)
    model:add(nn.Reshape(fmSz[1]*fmSz[1],dim))
    model:add(nn.Reshape(fmSz[1],fmSz[1],dim))
    model:add(nn.Transpose({3,4},{2,3}))

    return model
end

----------------------------
VGG_pretrain = loadcaffe.load("/media/sda1/Model/VGG/deploy.prototxt","/media/sda1/Model/VGG/VGG_ILSVRC_16_layers_fc_reduced.caffemodel","cudnn")

VGGNet = nn.Sequential()

for i = 1,23 do

    if i == 17 then
        VGGNet:add(nn.SpatialMaxPooling(2,2,2,2,1,1))
    else
        VGGNet:add(VGG_pretrain:get(i))
    end

end

subBranch_1 = nn.Sequential()

kernelSz = 3
prev_fDim = 512
next_fDim = 3*(classNum+4)
--subBranch_1:add(build_l2_normalize(512))
subBranch_1:add(nn.SpatialCrossMapLRN(1024,1024,0.5,0))
custom_cmul = nn.customCMul(1,512,1,1)
custom_cmul:reset(20/math.sqrt(512))
subBranch_1:add(custom_cmul)
subBranch_1:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))

------------------------------
mainBranch_1 = nn.Sequential()

for i = 24,36 do
    
    if i == 34 then
        goto dropout
    end

    if i == 31 then
        mainBranch_1:add(nn.SpatialMaxPooling(3,3,1,1,1,1))
    elseif i == 32 then
        kernelSz = 3
        prev_fDim = 512
        next_fDim = 1024
        dilatedStep = 3
        
        dilatedConvFromCaffe = VGG_pretrain:get(i)
        dilatedConvTorch = nn.normalDilatedConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,dilatedStep*(kernelSz-1)/2,dilatedStep*(kernelSz-1)/2,dilatedStep,dilatedStep,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim)))

        dilatedConvTorch.weight = dilatedConvFromCaffe.weight
        dilatedConvTorch.bias = dilatedConvFromCaffe.bias
        mainBranch_1:add(dilatedConvTorch)

    else
        mainBranch_1:add(VGG_pretrain:get(i))
    end

    ::dropout::
end

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

subBranch_6 = nn.Sequential()
kernelSz = 1
prev_fDim = 256
next_fDim = 5*(classNum+4)
subBranch_6:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
--------------------------------
model = nn.Sequential():add(VGGNet)

concat = nn.ConcatTable()
concat:add(subBranch_1)
concat:add(mainBranch_1)
model:add(concat)

concat = nn.ConcatTable()
concat:add(nn.SelectTable(1))
concat:add(nn.Sequential():add(nn.SelectTable(2)):add(subBranch_2))
concat:add(nn.Sequential():add(nn.SelectTable(2)):add(mainBranch_2))
model:add(concat)

concat = nn.ConcatTable()
concat:add(nn.SelectTable(1))
concat:add(nn.SelectTable(2))
concat:add(nn.Sequential():add(nn.SelectTable(3)):add(subBranch_3))
concat:add(nn.Sequential():add(nn.SelectTable(3)):add(mainBranch_3))
model:add(concat)

concat = nn.ConcatTable()
concat:add(nn.SelectTable(1))
concat:add(nn.SelectTable(2))
concat:add(nn.SelectTable(3))
concat:add(nn.Sequential():add(nn.SelectTable(4)):add(subBranch_4))
concat:add(nn.Sequential():add(nn.SelectTable(4)):add(mainBranch_4))
model:add(concat)

concat = nn.ConcatTable()
concat:add(nn.SelectTable(1))
concat:add(nn.SelectTable(2))
concat:add(nn.SelectTable(3))
concat:add(nn.SelectTable(4))
concat:add(nn.Sequential():add(nn.SelectTable(5)):add(subBranch_5))
concat:add(nn.Sequential():add(nn.SelectTable(5)):add(mainBranch_5))
model:add(concat)

concat = nn.ConcatTable()
concat:add(nn.SelectTable(1))
concat:add(nn.SelectTable(2))
concat:add(nn.SelectTable(3))
concat:add(nn.SelectTable(4))
concat:add(nn.SelectTable(5))
concat:add(nn.Sequential():add(nn.SelectTable(6)):add(subBranch_6))
model:add(concat)

---------------------------
crossEntropy = nn.CrossEntropyCriterion()
crossEntropy.nll.sizeAverage = false
smoothL1 = nn.SmoothL1Criterion()
smoothL1.sizeAverage = false
SM = nn.SoftMax()
SpatialLSM = cudnn.SpatialLogSoftMax()
SpatialSM = cudnn.SpatialSoftMax()

cudnn.convert(model, cudnn)
model:cuda()
crossEntropy:cuda()
smoothL1:cuda()
SpatialLSM:cuda()
SpatialSM:cuda()
SM:cuda()
cudnn.fastest = true
cudnn.benchmark = true
