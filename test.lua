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
    
    print("testing start!")

    os.execute('rm -f ' .. fig_dir .. '*') 
       
    model:evaluate()

    testDataSz = table.getn(testName)
    local startTime = sys.clock() 
    for t = 1,testDataSz do
        
        input = image.load(testName[t])
        
        imgWidth = input:size()[3]
        imgHeight = input:size()[2]

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
            
            local ar_num = lid2arnum(lid)

            for aid = 1,ar_num do 
            
                --conf thresholding
                local conf = output[lid][{{1},{(aid-1)*classNum+1,aid*classNum},{},{}}]
                conf = SpatialSM:forward(conf)
                conf = conf[1][{{1,classNum-1},{},{}}]

                conf,label = torch.max(conf,1)
                conf_mask = conf:gt(thr)
                conf_mask = conf_mask:type('torch.ByteTensor')
                
                conf = conf[conf_mask]:type('torch.FloatTensor')
                label = label[conf_mask]:type('torch.FloatTensor')
                local rest_box_num = torch.sum(conf_mask)

                local xmax = restored_box[lid][aid][1][conf_mask]
                local xmin = restored_box[lid][aid][2][conf_mask]
                local ymax = restored_box[lid][aid][3][conf_mask]
                local ymin = restored_box[lid][aid][4][conf_mask]
                
                --[===[
                --bb regression apply
                local loc_offset = output[lid][{{1},{ar_num*classNum+(aid-1)*4+1,ar_num*classNum+(aid-1)*4+4},{},{}}]
                local tx = loc_offset[1][1][conf_mask]:type('torch.FloatTensor')
                local ty = loc_offset[1][2][conf_mask]:type('torch.FloatTensor')
                local tw = loc_offset[1][3][conf_mask]:type('torch.FloatTensor')
                local th = loc_offset[1][4][conf_mask]:type('torch.FloatTensor')

                local newCenterX = torch.cmul(tx,(xmax-xmin+1)) + (xmax+xmin)/2
                local newCenterY = torch.cmul(ty,(ymax-ymin+1)) + (ymax+ymin)/2
                local newWidth = torch.cmul(torch.exp(tw),(xmax-xmin+1))
                local newHeight = torch.cmul(torch.exp(th),(ymax-ymin+1))
                
                xmax = torch.cmin(torch.ceil(newCenterX + newWidth/2 - 0.5),imgSz)
                xmin = torch.cmax(torch.ceil(newCenterX - newWidth/2 + 0.5),1)
                ymax = torch.cmin(torch.ceil(newCenterY + newHeight/2 - 0.5),imgSz)
                ymin = torch.cmax(torch.ceil(newCenterY - newHeight/2 + 0.5),1)
                --]===]

                --result save to table(before NMS)
                for rid = 1,rest_box_num do
                    if xmin[rid] < imgSz and ymin[rid] < imgSz and xmax[rid] > 1 and ymax[rid] > 1 then
                        table.insert(resultBB[label[rid]],{xmin[rid],ymin[rid],xmax[rid],ymax[rid],conf[rid]})
                    end
                end

            end
        end
        
        flag = false
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
                resultTensor = torch.cat(resultTensor,torch.Tensor(resultTensor:size()[1],1):fill(lid),2)
                
                if flag == false then
                    before_top_k = resultTensor
                    flag = true
                else
                    before_top_k = torch.cat(before_top_k,resultTensor,1)
                end
                
            end
        end
        
        --extract topk detection
        if before_top_k:size()[1] > topk_num then
            
            for lid = 1,classNum-1 do
                if type(resultBB[lid]) == 'userdata' then
                    resultBB[lid] = false
                end
            end

            after_top_k_val, after_top_k_idx = torch.reshape(before_top_k[{{},{5}}],before_top_k:size()[1]):topk(topk_num,true)

            for tid = 1,topk_num do
                local idx = after_top_k_idx[tid]
                local lid = before_top_k[idx][6]
                
                if type(resultBB[lid]) ~= "table" then
                    if type(resultBB[lid]) == 'boolean' then
                        resultBB[lid] = torch.reshape(before_top_k[idx],1,before_top_k[idx]:size()[1])
                    else
                        resultBB[lid] = torch.cat(resultBB[lid],torch.reshape(before_top_k[idx],1,before_top_k[idx]:size()[1])
,1)
                    end
                end

            end
        end


        --result write to txt file
        for lid = 1,classNum-1 do
            fp_result = io.open("comp3_det_val_" .. classList[lid] .. ".txt","a")
            if type(resultBB[lid]) == "userdata" then
                
                for rid = 1,resultBB[lid]:size()[1] do
                    
                    local xmax = resultBB[lid][rid][3]
                    local xmin = resultBB[lid][rid][1]
                    local ymax = resultBB[lid][rid][4]
                    local ymin = resultBB[lid][rid][2]

                    split_file_name = str_split(testName[t],"/")
                    split_file_name = split_file_name[table.getn(split_file_name)]
                    split_file_name = split_file_name:sub(1,-5)
                    
                    fp_result:write(split_file_name, " ", resultBB[lid][rid][5], " ", (xmin-1)*(imgWidth/imgSz)+1, " " , (ymin-1)*(imgHeight/imgSz)+1, " ", (xmax-1)*(imgWidth/imgSz)+1, " ", (ymax-1)*(imgHeight/imgSz)+1,"\n")

                                       
                    --input = drawRectangle(input,xmin,ymin,xmax,ymax,"r")

                end
            end
            fp_result:close()
        end

        --draw BB
        --image.save(tostring(t) .. ".jpg",input)
        
    end
    local endTime = sys.clock()
    print("fps: " .. tostring(testDataSz/(endTime-startTime)))
   

end
