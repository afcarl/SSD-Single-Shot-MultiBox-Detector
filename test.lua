require 'torch'
require 'xlua' 
require 'optim'
require 'image'
require 'os'
require 'sys'

-- Original author: Francisco Massa: https://github.com/fmassa/object-detection.torch 
-- Based on matlab code by Pedro Felzenszwalb https://github.com/rbgirshick/voc-dpm/blob/master/test/nms.m
-- Minor changes by Gyeongsik Moon(2016-10-03) 
function NMS(boxes, overlap, scores)

    local pick = torch.FloatTensor(boxes:size()[1]):zero()


  local x1 = boxes[{{}, 1}]
  local y1 = boxes[{{}, 2}]
  local x2 = boxes[{{}, 3}]
  local y2 = boxes[{{}, 4}]
    
  local area = torch.cmul(x2 - x1 + 1, y2 - y1 + 1)
  
  local v, I = scores:sort(1)
  local count = 1
  
  local xx1 = boxes.new()
  local yy1 = boxes.new()
  local xx2 = boxes.new()
  local yy2 = boxes.new()

  local w = boxes.new()
  local h = boxes.new()

  while I:numel() > 0 do 
    local last = I:size(1)
    local i = I[last]
    
    pick[i] = 1
    count = count + 1
    
    if last == 1 then
      break
    end
    
    I = I[{{1, last-1}}] -- remove picked element from view
    
    -- load values 
    xx1:index(x1, 1, I)
    yy1:index(y1, 1, I)
    xx2:index(x2, 1, I)
    yy2:index(y2, 1, I)
    
    -- compute intersection area
    xx1:cmax(x1[i])
    yy1:cmax(y1[i])
    xx2:cmin(x2[i])
    yy2:cmin(y2[i])
    
    w:resizeAs(xx2)
    h:resizeAs(yy2)
    torch.add(w, xx2, -1, xx1):add(1):cmax(0)
    torch.add(h, yy2, -1, yy1):add(1):cmax(0)
    
    -- reuse existing tensors
    local inter = w:cmul(h)
    local IoU = h
    
    -- IoU := i / (area(a) + area(b) - i)
    xx1:index(area, 1, I) -- load remaining areas into xx1
    torch.cdiv(IoU, inter, xx1 + area[i] - inter) -- store result in iou
    
    I = I[IoU:le(overlap)] -- keep only elements with a IoU < overlap 
  end
    
  pick = torch.reshape(pick,pick:size()[1],1):repeatTensor(1,5)

  return pick
end

function test(testTarget, testName)
    
    os.execute('rm -f ' .. fig_dir .. '*') 
    os.execute('rm -f ' .. score_dir .. '*') 
  
    model:evaluate()

    testDataSz = table.getn(testName)
    local startTime = sys.clock() 
    for t = 1,testDataSz do
        
        print(tostring(t) .. "/" .. tostring(testDataSz) ..  "th frame is in testing...")

        input = image.load(testName[t])
        originSz = input:size()
        input = image.scale(input,imgSz,imgSz)
        target = testTarget[t]
          
        input = input:cuda()
        input = torch.reshape(input,1,inputDim,imgSz,imgSz)
                            
        local output = model:forward(input)
        
        input = torch.reshape(input,inputDim,imgSz,imgSz)
        resultBB = {}
        for lid = 1,classNum-1 do
            table.insert(resultBB,{})
        end

        for lid = 1,m do
            
            if lid == 1 then
                ar_num = 3
            elseif lid < m then
                ar_num = 6
            else
                ar_num = 5
            end

            for aid = 1,ar_num do 
            
                --conf thresholding
                local conf = output[lid][{{1},{(aid-1)*classNum+1,aid*classNum},{},{}}]
                conf = nn.SpatialSoftMax():cuda():forward(conf)
                conf = conf[1]

                conf,label = torch.max(conf,1)
                conf_mask = torch.cmul(conf:gt(thr),label:ne(21))
                conf_mask = conf_mask:type('torch.ByteTensor')
                
                conf = conf[conf_mask]:type('torch.FloatTensor')
                label = label[conf_mask]:type('torch.FloatTensor')
                local rest_box_num = torch.sum(conf_mask)

                local xmax = restored_box[lid][aid][1][conf_mask]
                local xmin = restored_box[lid][aid][2][conf_mask]
                local ymax = restored_box[lid][aid][3][conf_mask]
                local ymin = restored_box[lid][aid][4][conf_mask]
                
                --bb regression apply
                local loc_offset = output[lid][{{1},{ar_num*classNum+(aid-1)*4+1,ar_num*classNum+(aid-1)*4+4},{},{}}]
                local tx = loc_offset[1][1][conf_mask]:type('torch.FloatTensor')
                local ty = loc_offset[1][2][conf_mask]:type('torch.FloatTensor')
                local tw = loc_offset[1][3][conf_mask]:type('torch.FloatTensor')
                local th = loc_offset[1][4][conf_mask]:type('torch.FloatTensor')

                local newCenterX = torch.cmul(tx,(xmax-xmin)) + (xmax+xmin)/2
                local newCenterY = torch.cmul(ty,(ymax-ymin)) + (ymax+ymin)/2
                local newWidth = torch.cmul(torch.exp(tw),(xmax-xmin))
                local newHeight = torch.cmul(torch.exp(th),(ymax-ymin))
                
                xmax = torch.cmin(newCenterX + newWidth/2,torch.Tensor(rest_box_num):fill(imgSz))
                xmin = torch.cmax(newCenterX - newWidth/2,torch.Tensor(rest_box_num):fill(1))
                ymax = torch.cmin(newCenterY + newHeight/2,torch.Tensor(rest_box_num):fill(imgSz))
                ymin = torch.cmax(newCenterY - newHeight/2,torch.Tensor(rest_box_num):fill(1))
                
                --result save to table(before NMS)
                for rid = 1,rest_box_num do
                    table.insert(resultBB[label[rid]],{xmin[rid],ymin[rid],xmax[rid],ymax[rid],conf[rid]})
                end

            end
        end
        
        
        --NMS for each class
        for lid = 1,classNum-1 do
            
            if table.getn(resultBB[lid]) > 0 then

                local resultTensor = torch.Tensor(resultBB[lid])
                local box = resultTensor[{{},{1,4}}]
                local score = torch.reshape(resultTensor[{{},{5}}],resultTensor:size()[1])

                idx = NMS(box,0.45,score)
                resultTensor = resultTensor[idx:type('torch.ByteTensor')] 
                resultTensor = torch.reshape(resultTensor,resultTensor:size()[1]/5,5)
                resultBB[lid] = resultTensor
            end
        end 
          

        --result write to txt file
        for lid = 1,classNum-1 do
            fp_result = io.open(result_dir .. "score/comp3_det_test_" .. classList[lid] .. ".txt","a")
            if type(resultBB[lid]) == "userdata" then

                for rid = 1,resultBB[lid]:size()[1] do
                    
                    local xmax = resultBB[lid][rid][3]
                    local xmin = resultBB[lid][rid][1]
                    local ymax = resultBB[lid][rid][4]
                    local ymin = resultBB[lid][rid][2]

                    split_file_name = str_split(testName[t],"/")
                    split_file_name = split_file_name[table.getn(split_file_name)]
                    split_file_name = split_file_name:sub(1,-5)
                    
                    fp_result:write(split_file_name, " ", resultBB[lid][rid][5], " ", xmin/imgSz*originSz[3], " " , ymin/imgSz*originSz[2], " ", xmax/imgSz*originSz[3], " ", ymax/imgSz*originSz[2],"\n")

                                       
                    input = drawRectangle(input,xmin,ymin,xmax,ymax)

                end
            end
            fp_result:close()
        end


        --draw BB
        image.save(fig_dir .. tostring(t) .. ".jpg",input)
        
    end
    local endTime = sys.clock()
    print("fps: " .. tostring(testDataSz/(endTime-startTime)))
   

end
