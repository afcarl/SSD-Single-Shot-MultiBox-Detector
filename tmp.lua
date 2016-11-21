require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'module/normalConv'
require 'module/normalLinear'
require 'sys'
dofile 'etc.lua'
dofile 'data.lua'

testTarget, testName = load_data("test")

local input = image.load(testName[1])
input = image.scale(input,imgSz,imgSz)

--for debug
local img = input
local xmax = restored_box[1][4][1][19][19]
local xmin = restored_box[1][4][2][19][19]
local ymax = restored_box[1][4][3][19][19]
local ymin = restored_box[1][4][4][19][19]
img = drawRectangle(img,xmin,ymin,xmax,ymax,"r")
image.save("1.jpg",img)

