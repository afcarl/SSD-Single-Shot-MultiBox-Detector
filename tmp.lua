require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'module/normalConv'
require 'module/normalLinear'
require 'sys'
dofile 'etc.lua'


for line in io.lines("/home/mks0601/workspace/Data/VOCdevkit/VOC2012_trainval/ImageSets/Main/val.txt") do
    print(line)
end
