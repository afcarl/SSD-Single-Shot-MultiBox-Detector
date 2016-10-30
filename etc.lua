db_dir = "/home/mks0601/workspace/Data/VOCdevkit/"
result_dir = "/home/mks0601/workspace/Data/VOCdevkit/results/VOC2012/Main/"
model_dir = result_dir .. "model/"
fig_dir = result_dir .. "fig/"
score_dir = result_dir .. "score/"

mode = "test"
continue = false
continue_iter = 0

classNum = 21
inputDim = 3
imgSz = 500
trainSz = 17125 + 5011 + 4952
thr = 0.4
classList = {"aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"}

m = 6
scale_table = {}
for k=1,m do
    table.insert(scale_table,0.2 + (0.95 - 0.2)/(m-1) * (k-1))
end
ar_table = {1,2,1/2,3,1/3}
fmSz = {32,16,8,4,2,1}

lr = 1e-3
wDecay = 5e-4
mmt = 9e-1
batchSz = 8
iterLimit = 6e4 - continue_iter
iterLrDecay = 4e4 - continue_iter

function str_split(inputstr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={} ; i=1
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
                t[i] = str
                i = i + 1
        end
        return t
end
  
restored_box = {} --xmax xmin ymax ymin
table.insert(restored_box,torch.Tensor(6,4,fmSz[1],fmSz[1]):zero())
table.insert(restored_box,torch.Tensor(6,4,fmSz[2],fmSz[2]):zero())
table.insert(restored_box,torch.Tensor(6,4,fmSz[3],fmSz[3]):zero())
table.insert(restored_box,torch.Tensor(6,4,fmSz[4],fmSz[4]):zero())
table.insert(restored_box,torch.Tensor(6,4,fmSz[5],fmSz[5]):zero())
table.insert(restored_box,torch.Tensor(5,4,fmSz[6],fmSz[6]):zero())

for lid = 1,m do
       
   for r = 1,fmSz[lid] do
       for c = 1,fmSz[lid] do
            local xCenter = (c-1+0.5)/fmSz[lid]
            local yCenter = (r-1+0.5)/fmSz[lid]

            local ar_num = table.getn(ar_table)+1
            if lid == m then
                ar_num = table.getn(ar_table)
            end

            for aid = 1,ar_num do
                if lid == m then --last layer
                    ar_factor = ar_table[aid]
                    scale_factor = scale_table[lid]

                elseif lid < m then --before last layer
                    if aid <= table.getn(ar_table) then
                        ar_factor = ar_table[aid]
                        scale_factor = scale_table[lid]
                    else
                        ar_factor = 1
                        scale_factor = math.sqrt(scale_table[lid] * scale_table[lid+1])
                    end
                end

                local width = scale_factor*math.sqrt(ar_factor)
                local height = scale_factor/math.sqrt(ar_factor)

                restored_box[lid][aid][1][r][c] = math.min((xCenter + width/2) * (imgSz),imgSz)
                restored_box[lid][aid][2][r][c] = math.max((xCenter - width/2) * (imgSz),1)
                restored_box[lid][aid][3][r][c] = math.min((yCenter + height/2) * (imgSz),imgSz)
                restored_box[lid][aid][4][r][c] = math.max((yCenter - height/2) * (imgSz),1)

                ::nextCell::
            end
        end
    end
end

function drawRectangle(img,xmin,ymin,xmax,ymax)
    
    img_origin = img:clone()
    img[1][{{ymin,ymax},{xmin,xmax}}] = 255
    img[2][{{ymin,ymax},{xmin,xmax}}] = 0
    img[3][{{ymin,ymax},{xmin,xmax}}] = 0
    
    if ymin+2 < ymax-2 then
        ymin = ymin+2
        ymax = ymax-2
    end

    if xmin+2 < xmax-2 then
        xmin = xmin+2
        xmax = xmax-2
    end

    img[1][{{ymin,ymax},{xmin,xmax}}] = img_origin[1][{{ymin,ymax},{xmin,xmax}}]
    img[2][{{ymin,ymax},{xmin,xmax}}] = img_origin[2][{{ymin,ymax},{xmin,xmax}}]
    img[3][{{ymin,ymax},{xmin,xmax}}] = img_origin[3][{{ymin,ymax},{xmin,xmax}}]

    return img
end


