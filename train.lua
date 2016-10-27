require 'torch'
require 'nn'
require 'optim'
require 'xlua'
require 'image'
require 'sys'
dofile 'etc.lua'

params, gradParams = model:getParameters()

optimState = {
    learningRate = lr,
    learningRateDecay = 0.0,
    weightDecay = wDecay,
    momentum = mmt,
}
optimMethod = optim.sgd
tot_error = 0
tot_cls_err = 0
tot_loc_err = 0
cnt_error = 0
tot_iter = 0


function train(trainTarget, trainName)
    
    
    tot_error = 0
    tot_cls_err = 0
    tot_loc_err = 0
    cnt_error = 0
    

    model:training()

    shuffle = torch.randperm(trainSz)
     
    for t = 1,trainSz,batchSz do
        if t+batchSz-1 <= trainSz then
            inputs = torch.CudaTensor(batchSz,inputDim,imgSz,imgSz)
            targets = {}
            curBatchDim = batchSz
        else
            inputs = torch.CudaTensor(trainSz-t+1,inputDim,imgSz,imgSz)
            targets = {}
            curBatchDim = trainSz-t+1
        end

        for i = t,math.min(t+batchSz-1,trainSz) do
            
            local input_name = trainName[shuffle[i]]
            local target = trainTarget[shuffle[i]]

            local input = image.load(input_name)
            input = image.scale(input,imgSz,imgSz)
            
            inputs[i-t+1] = input
            table.insert(targets,target)
        end
        

            local feval = function(x)
                    if x ~= params then
                        params:copy(x)
                    end

                    gradParams:zero()
                    class_error = 0
                    loc_error = 0
                   
                    doBP = {}
                    table.insert(doBP,torch.Tensor(curBatchDim,3,fmSz[1],fmSz[1]):zero())
                    table.insert(doBP,torch.Tensor(curBatchDim,6,fmSz[2],fmSz[2]):zero())
                    table.insert(doBP,torch.Tensor(curBatchDim,6,fmSz[3],fmSz[3]):zero())
                    table.insert(doBP,torch.Tensor(curBatchDim,6,fmSz[4],fmSz[4]):zero())
                    table.insert(doBP,torch.Tensor(curBatchDim,6,fmSz[5],fmSz[5]):zero())
                    table.insert(doBP,torch.Tensor(curBatchDim,5,fmSz[6],fmSz[6]):zero())

                     --layer, batch, ar*classNum + ar*4, sz, sz
                    local outputs = model:forward(inputs)
                    tot_dfdo = {}

                    for bid = 1,curBatchDim do
                        

                        local target = targets[bid]
                        local pos_set = {}
                        local neg_set = {}
                        local neg_candidate = {}
                        

                        for gid = 1,table.getn(target) do
                            
                            local pos_candidate = {}
                            local max_pose_score = -9999

                            local label = target[gid][1]
                            local xmax = target[gid][2]
                            local xmin = target[gid][3]
                            local ymax = target[gid][4]
                            local ymin = target[gid][5]
                            local imgWidth = target[gid][6]
                            local imgHeight = target[gid][7]
 
                            local xmax_ = xmax * (imgSz/imgWidth)
                            local xmin_ = xmin * (imgSz/imgWidth)
                            local ymax_ = ymax * (imgSz/imgHeight)
                            local ymin_ = ymin * (imgSz/imgHeight)
                           
                            
                            --for debug
                            local img = inputs[bid]
                            img = drawRectangle(img,xmin_,ymin_,xmax_,ymax_)
                            image.save(tostring(bid) .. "_" .. tostring(gid) .. ".jpg",img)
                            
                            local gt_area = (xmax_ - xmin_) * (ymax_ - ymin_)
                            
                            for lid = 1,m do
                                
                                if lid == 1 then
                                    ar_num = 3
                                elseif lid < m then
                                    ar_num = 6
                                else
                                    ar_num = 5
                                end
                                
                                --assign one box to each GT(best match box)
                                local minXMax = torch.cmin(torch.Tensor(ar_num,fmSz[lid],fmSz[lid],1):fill(xmax_),restored_box[lid][{{},{1},{},{}}])
                                local maxXMin = torch.cmax(torch.Tensor(ar_num,fmSz[lid],fmSz[lid],1):fill(xmin_),restored_box[lid][{{},{2},{},{}}])
                                local minYMAX = torch.cmin(torch.Tensor(ar_num,fmSz[lid],fmSz[lid],1):fill(ymax_),restored_box[lid][{{},{3},{},{}}])
                                local maxYMIN = torch.cmax(torch.Tensor(ar_num,fmSz[lid],fmSz[lid],1):fill(ymin_),restored_box[lid][{{},{4},{},{}}])
                                local box_area = torch.cmul((restored_box[lid][{{},{1},{},{}}] - restored_box[lid][{{},{2},{},{}}]), (restored_box[lid][{{},{3},{},{}}] - restored_box[lid][{{},{4},{},{}}]))

                                local area_inter = torch.cmul(torch.cmax(minXMax - maxXMin,0), torch.cmax(minYMAX - maxYMIN,0))
                                local area_union = torch.Tensor(ar_num,fmSz[lid],fmSz[lid],1):fill(gt_area) + box_area - area_inter
                                local IoU = torch.cdiv(area_inter,area_union)
                                IoU = torch.reshape(IoU,ar_num,fmSz[lid],fmSz[lid])

                                local val_1,arIdx = torch.max(IoU,1)
                                local val_2,yIdx = torch.max(val_1,2)
                                local val_3,xIdx = torch.max(val_2,3)
                                xIdx = xIdx[1][1][1]
                                yIdx = yIdx[1][1][xIdx]
                                arIdx = arIdx[1][yIdx][xIdx]
                                
                                
                                --for debug
                                local img = inputs[bid]
                                local xmax = restored_box[lid][arIdx][1][yIdx][xIdx]
                                local xmin = restored_box[lid][arIdx][2][yIdx][xIdx]
                                local ymax = restored_box[lid][arIdx][3][yIdx][xIdx]
                                local ymin = restored_box[lid][arIdx][4][yIdx][xIdx]
                                img = drawRectangle(img,xmin,ymin,xmax,ymax)
                                image.save(tostring(bid) .. "_" .. tostring(gid) .. "_" .. tostring(lid) .. ".jpg",img)
                                
                                local tx = ((xmin_+xmax_)/2 - (restored_box[lid][arIdx][2][yIdx][xIdx]+restored_box[lid][arIdx][1][yIdx][xIdx])/2)/(restored_box[lid][arIdx][1][yIdx][xIdx]-restored_box[lid][arIdx][2][yIdx][xIdx])
                                local ty = ((ymin_+ymax_)/2 - (restored_box[lid][arIdx][4][yIdx][xIdx]+restored_box[lid][arIdx][3][yIdx][xIdx])/2)/(restored_box[lid][arIdx][3][yIdx][xIdx]-restored_box[lid][arIdx][4][yIdx][xIdx])
                                local tw = math.log((xmax_-xmin_)/(restored_box[lid][arIdx][1][yIdx][xIdx]-restored_box[lid][arIdx][2][yIdx][xIdx]))
                                local th = math.log((ymax_-ymin_)/(restored_box[lid][arIdx][3][yIdx][xIdx]-restored_box[lid][arIdx][4][yIdx][xIdx]))
                                
                                if IoU[arIdx][yIdx][xIdx] > max_pose_score then
                                    pos_candidate = {lid,arIdx,yIdx,xIdx,label,tx,ty,tw,th,IoU[arIdx][yIdx][xIdx]}
                                    max_pose_score = IoU[arIdx][yIdx][xIdx]
                                end
                                
                                --assign boxes whose IoU > 0.5
                                local IoU_cut = torch.cmul(IoU:gt(0.5),doBP[lid][bid]:ne(1))
                                local IoU_cut_num = torch.sum(IoU_cut)
                                
                                local acoord = torch.reshape(torch.range(1,ar_num),ar_num,1,1):repeatTensor(1,fmSz[lid],fmSz[lid])
                                local xcoord = torch.range(1,fmSz[lid]):repeatTensor(ar_num,fmSz[lid],1)
                                local ycoord = xcoord:transpose(2,3)
                                acoord = acoord[IoU_cut]
                                xcoord = xcoord[IoU_cut]
                                ycoord = ycoord[IoU_cut]
                                
                                for pid = 1,IoU_cut_num do
                                    local arIdx = acoord[pid]
                                    local yIdx = ycoord[pid]
                                    local xIdx = xcoord[pid]
                                    local tx = ((xmin_+xmax_)/2 - (restored_box[lid][arIdx][2][yIdx][xIdx]+restored_box[lid][arIdx][1][yIdx][xIdx])/2)/(restored_box[lid][arIdx][1][yIdx][xIdx]-restored_box[lid][arIdx][2][yIdx][xIdx])
                                    local ty = ((ymin_+ymax_)/2 - (restored_box[lid][arIdx][4][yIdx][xIdx]+restored_box[lid][arIdx][3][yIdx][xIdx])/2)/(restored_box[lid][arIdx][3][yIdx][xIdx]-restored_box[lid][arIdx][4][yIdx][xIdx])
                                    local tw = math.log((xmax_-xmin_)/(restored_box[lid][arIdx][1][yIdx][xIdx]-restored_box[lid][arIdx][2][yIdx][xIdx]))
                                    local th = math.log((ymax_-ymin_)/(restored_box[lid][arIdx][3][yIdx][xIdx]-restored_box[lid][arIdx][4][yIdx][xIdx]))


                                    table.insert(pos_set,{lid,acoord[pid],ycoord[pid],xcoord[pid],label,tx,ty,tw,th})
                                    doBP[lid][bid][arIdx][yIdx][xIdx] = 1
                                end

                                                                
                                --hard neg mining
                                for aid = 1,ar_num do
                                    
                                    local IoU_cut = IoU[aid]:lt(0.2)
                                    local IoU_cut_num = torch.sum(IoU_cut)

                                    if IoU_cut_num > 0 then
                                         
                                        local conf_1, arIdx = torch.max(outputs[lid][bid][{{(aid-1)*classNum+1,aid*classNum-1},{},{}}],1)
                                        conf_1[1][IoU_cut:eq(0)] = -100
                                        local conf_2, yIdx = torch.max(conf_1,2)
                                        local conf_3, xIdx = torch.max(conf_2,3)

                                        xIdx = xIdx[1][1][1]
                                        yIdx = yIdx[1][1][xIdx]
                                        arIdx = arIdx[1][yIdx][xIdx]
                                        
                                        table.insert(neg_candidate,{lid,aid,yIdx,xIdx,21,outputs[lid][bid][(aid-1)*classNum+arIdx][yIdx][xIdx]})
                                    end
                                
                                end
                                
                            end

                            --pos assign
                            table.insert(pos_set,pos_candidate)
                            local lid = pos_candidate[1]
                            local arIdx = pos_candidate[2]
                            local yIdx = pos_candidate[3]
                            local xIdx = pos_candidate[4]
                            doBP[lid][bid][arIdx][yIdx][xIdx] = 1

                        end

                        

                        --hard neg assign
                        local neg_num = math.min(table.getn(neg_candidate),3*table.getn(target))
                        function compare(a,b)
                            return a[6] > b[6]
                        end
                        table.sort(neg_candidate,compare)
                        for nid = 1,neg_num do
                            local lid = neg_candidate[nid][1]
                            local aid = neg_candidate[nid][2]
                            local y = neg_candidate[nid][3]
                            local x = neg_candidate[nid][4]
                            local label = neg_candidate[nid][5]

                            doBP[lid][bid][aid][y][x] = -1
                            table.insert(neg_set,{lid,aid,y,x,label})
                        end
                    
                        
                        --final sum up gradient 
                        --class gradient
                        local classOutput = torch.Tensor(table.getn(pos_set)+table.getn(neg_set),classNum)
                        local classGT = torch.Tensor(table.getn(pos_set)+table.getn(neg_set),1)
                        local locOutput = torch.Tensor(table.getn(pos_set),4)
                        local locGT = torch.Tensor(table.getn(pos_set),4)
                        
                        for pid = 1,table.getn(pos_set) do
                            
                            local lid = pos_set[pid][1]
                            local aid = pos_set[pid][2]
                            local yid = pos_set[pid][3]
                            local xid = pos_set[pid][4]
                            local label = pos_set[pid][5]
                                                       
                            classOutput[pid] = outputs[lid][bid][{{(aid-1)*classNum+1,aid*classNum},{yid},{xid}}]:type('torch.FloatTensor')
                            classGT[pid] = label

                            
                            local tx = pos_set[pid][6]
                            local ty = pos_set[pid][7]
                            local tw = pos_set[pid][8]
                            local th = pos_set[pid][9]
                            
                            --[===[
                            --for debug
                            local img = inputs[bid]
                            local xmax = restored_box[lid][aid][1][yid][xid]
                            local xmin = restored_box[lid][aid][2][yid][xid]
                            local ymax = restored_box[lid][aid][3][yid][xid]
                            local ymin = restored_box[lid][aid][4][yid][xid]
                            newCenterX = tx*(xmax-xmin) + (xmax+xmin)/2
                            newCenterY = ty*(ymax-ymin) + (ymax+ymin)/2
                            newWidth = math.exp(tw)*(xmax-xmin)
                            newHeight = math.exp(th)*(ymax-ymin)
                            xmax = newCenterX + newWidth/2
                            xmin = newCenterX - newWidth/2
                            ymax = newCenterY + newHeight/2
                            ymin = newCenterY - newHeight/2
                            img = drawRectangle(img,xmin,ymin,xmax,ymax)
                            image.save(tostring(label) .. "_" .. tostring(bid) .. "_" .. tostring(pid) .. ".jpg",img)
                            --]===]
                            
                            if lid == 1 then
                                ar_num = 3
                            elseif lid < m then
                                ar_num = 6
                            else
                                ar_num = 5
                            end

                            locOutput[pid] = outputs[lid][bid][{{ar_num*classNum + (aid-1)*4+1,ar_num*classNum + (aid-1)*4+4},{yid},{xid}}]:type('torch.FloatTensor')
                            locGT[pid] = torch.Tensor({tx,ty,tw,th})

                        end

                        for nid = 1,table.getn(neg_set) do
                            local lid = neg_set[nid][1]
                            local aid = neg_set[nid][2]
                            local yid = neg_set[nid][3]
                            local xid = neg_set[nid][4]
                            local label = neg_set[nid][5]

                            classOutput[table.getn(pos_set)+nid] = outputs[lid][bid][{{(aid-1)*classNum+1,aid*classNum},{yid},{xid}}]:type('torch.FloatTensor')
                            classGT[table.getn(pos_set)+nid] = label
                        end
                        
                        classOutput = classOutput:cuda()
                        classGT = classGT:cuda()
                        class_error = class_error + crossEntropy:forward(classOutput,classGT)
                        class_dfdo = crossEntropy:backward(classOutput,classGT)
                        
                                                
                        locOutput = locOutput:cuda()
                        locGT = locGT:cuda()
                        loc_error = loc_error + smoothL1:forward(locOutput,locGT)/table.getn(pos_set)
                        loc_dfdo = smoothL1:backward(locOutput,locGT)/table.getn(pos_set)

                        if bid == 1 then
                            for lid = 1,m do
                                local dfdo = torch.CudaTensor(curBatchDim,outputs[lid]:size()[2],fmSz[lid],fmSz[lid]):zero()
                                table.insert(tot_dfdo,dfdo)
                            end
                        end


                        for pid = 1,table.getn(pos_set) do

                            local lid = pos_set[pid][1]
                            local aid = pos_set[pid][2]
                            local yid = pos_set[pid][3]
                            local xid = pos_set[pid][4]
                            local class_grad = class_dfdo[pid]
                            local loc_grad = loc_dfdo[pid]
                            
                            if lid == 1 then
                                ar_num = 3
                            elseif lid < m then
                                ar_num = 6
                            else
                                ar_num = 5
                            end

                            --class grad
                            tot_dfdo[lid][bid][{{(aid-1)*classNum+1,aid*classNum},{yid},{xid}}] = tot_dfdo[lid][bid][{{(aid-1)*classNum+1,aid*classNum},{yid},{xid}}] + class_grad
                            --loc grad
                            tot_dfdo[lid][bid][{{ar_num*classNum+(aid-1)*4+1,ar_num*classNum+(aid-1)*4+4},{yid},{xid}}] = tot_dfdo[lid][bid][{{ar_num*classNum+(aid-1)*4+1,ar_num*classNum+(aid-1)*4+4},{yid},{xid}}] + loc_grad
                        end

                        
                        for nid = 1,table.getn(neg_set) do

                            local lid = neg_set[nid][1]
                            local aid = neg_set[nid][2]
                            local yid = neg_set[nid][3]
                            local xid = neg_set[nid][4]
                            local class_grad = class_dfdo[table.getn(pos_set) + nid]
                            
                            if lid == 1 then
                                ar_num = 3
                            elseif lid < m then
                                ar_num = 6
                            else
                                ar_num = 5
                            end

                            --class grad
                            tot_dfdo[lid][bid][{{(aid-1)*classNum+1,aid*classNum},{yid},{xid}}] = tot_dfdo[lid][bid][{{(aid-1)*classNum+1,aid*classNum},{yid},{xid}}] + class_grad
                        end


                    end
                    
                    model:backward(inputs,tot_dfdo)

                    gradParams:div(curBatchDim)
                    class_error = class_error/curBatchDim
                    loc_error = loc_error/curBatchDim

                    err = class_error + loc_error
                    tot_error = tot_error + err
                    tot_cls_err = tot_cls_err + class_error
                    tot_loc_err = tot_loc_err + loc_error
                    cnt_error = cnt_error + 1
                    
                    return err,gradParams

                    end

         optimMethod(feval, params, optimState)
        
        tot_iter = tot_iter + 1

        if tot_iter % 100 == 0 then
            print("iteration: " .. tot_iter .. "/" .. iterLimit .. " batch: " ..  t .. "/" .. trainSz .. " loss: " .. tot_error/cnt_error .. " classErr: " .. tot_cls_err/cnt_error .. " locErr: " .. tot_loc_err/cnt_error)
        end

        if tot_iter == iterLrDecay then
            optimState.learningRate = optimState.learningRate/10
        end
        
    end
    
    local filename = paths.concat(model_dir, 'model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)
    model:clearState()
    torch.save(filename, model)
    
    fp_err = io.open(resultDir .. "/loss.txt","a")
    local err = tot_error/cnt_error
    fp_err:write(err,"\n")
    fp_err:close()

end



