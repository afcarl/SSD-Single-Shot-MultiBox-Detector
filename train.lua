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
pos_confusion = optim.ConfusionMatrix(classList_object)
neg_confusion = optim.ConfusionMatrix(classList)
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
                    
                    final_pos_conf_output = {}
                    final_pos_conf_target = {}
                    final_neg_conf_output = {}
                    final_neg_conf_target = {}
                    final_loc_output = {}
                    final_loc_target = {}
                    
                    tot_dfdo = {}
                    table.insert(tot_dfdo,torch.CudaTensor(curBatchDim,4*(4+classNum),fmSz[1],fmSz[1]):zero())
                    table.insert(tot_dfdo,torch.CudaTensor(curBatchDim,6*(4+classNum),fmSz[2],fmSz[2]):zero())
                    table.insert(tot_dfdo,torch.CudaTensor(curBatchDim,6*(4+classNum),fmSz[3],fmSz[3]):zero())
                    table.insert(tot_dfdo,torch.CudaTensor(curBatchDim,6*(4+classNum),fmSz[4],fmSz[4]):zero())
                    table.insert(tot_dfdo,torch.CudaTensor(curBatchDim,6*(4+classNum),fmSz[5],fmSz[5]):zero())
                    table.insert(tot_dfdo,torch.CudaTensor(curBatchDim,5*(4+classNum),fmSz[6],fmSz[6]):zero())


                    neg_candidate_loss = torch.Tensor(tot_box_num):zero()
                    pos_candidate_iou = torch.Tensor(tot_box_num):zero()
                    pos_candidate_label = torch.Tensor(tot_box_num):zero()
                    pos_candidate_xmax = torch.Tensor(tot_box_num):zero()
                    pos_candidate_xmin = torch.Tensor(tot_box_num):zero()
                    pos_candidate_ymax = torch.Tensor(tot_box_num):zero()
                    pos_candidate_ymin = torch.Tensor(tot_box_num):zero()

                    idx_tensor = torch.range(1,tot_box_num)

                    --layer, batch, ar*classNum + ar*4, sz, sz
                    local outputs = model:forward(inputs)
                    

                    for bid = 1,curBatchDim do

                        pos_candidate_iou:zero()

                        local target = targets[bid]
                        local pos_set = {}
                        local neg_set = {}
                        

                        for gid = 1,table.getn(target) do
                            
                            local pos_best_match = {}
                            local max_pos_iou = -9999

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

                            --image.save(classList[label] .. tostring(bid) .. ".jpg",inputs[bid][{{},{ymin_,ymax_},{xmin_,xmax_}}])

                            
                            --[===[
                            --for debug
                            local img = inputs[bid]
                            img = drawRectangle(img,xmin_,ymin_,xmax_,ymax_,"r")
                            image.save(tostring(bid) .. "_" .. tostring(gid) .. ".jpg",img)
                            --]===]
                            
                            local gt_area = (xmax_ - xmin_) * (ymax_ - ymin_)
                            local startIdx = 1

                            for lid = 1,m do
                                
                                if lid == 1 then
                                    ar_num = 4
                                elseif lid < m then
                                    ar_num = 6
                                else
                                    ar_num = 5
                                end
                                
                                --assign one box to each GT(best match box)
                                local minXMax = torch.cmin(torch.Tensor(ar_num,1,fmSz[lid],fmSz[lid]):fill(xmax_),restored_box[lid][{{},{1},{},{}}])
                                local maxXMin = torch.cmax(torch.Tensor(ar_num,1,fmSz[lid],fmSz[lid]):fill(xmin_),restored_box[lid][{{},{2},{},{}}])
                                local minYMAX = torch.cmin(torch.Tensor(ar_num,1,fmSz[lid],fmSz[lid]):fill(ymax_),restored_box[lid][{{},{3},{},{}}])
                                local maxYMIN = torch.cmax(torch.Tensor(ar_num,1,fmSz[lid],fmSz[lid]):fill(ymin_),restored_box[lid][{{},{4},{},{}}])
                                local box_area = torch.cmul((restored_box[lid][{{},{1},{},{}}] - restored_box[lid][{{},{2},{},{}}]), (restored_box[lid][{{},{3},{},{}}] - restored_box[lid][{{},{4},{},{}}]))

                                local area_inter = torch.cmul(torch.cmax(minXMax - maxXMin,0), torch.cmax(minYMAX - maxYMIN,0))
                                local area_union = torch.Tensor(ar_num,1,fmSz[lid],fmSz[lid]):fill(gt_area) + box_area - area_inter
                                local IoU = torch.cdiv(area_inter,area_union)
                                IoU = torch.reshape(IoU,ar_num,fmSz[lid],fmSz[lid])

                                local val_1,aid = torch.max(IoU,1)
                                local val_2,yid = torch.max(val_1,2)
                                local val_3,xid = torch.max(val_2,3)
                                xid = xid[1][1][1]
                                yid = yid[1][1][xid]
                                aid = aid[1][yid][xid]
                                
                                --[===[
                                --for debug
                                local img = inputs[bid]
                                local xmax = restored_box[lid][aid][1][yid][xid]
                                local xmin = restored_box[lid][aid][2][yid][xid]
                                local ymax = restored_box[lid][aid][3][yid][xid]
                                local ymin = restored_box[lid][aid][4][yid][xid]
                                img = drawRectangle(img,xmin,ymin,xmax,ymax,"r")
                                image.save(tostring(bid) .. "_" .. tostring(gid) .. "_" .. tostring(lid) .. ".jpg",img)
                                --]===]
                                
                                local tx = ((xmin_+xmax_)/2 - (restored_box[lid][aid][2][yid][xid]+restored_box[lid][aid][1][yid][xid])/2)/(restored_box[lid][aid][1][yid][xid]-restored_box[lid][aid][2][yid][xid])
                                local ty = ((ymin_+ymax_)/2 - (restored_box[lid][aid][4][yid][xid]+restored_box[lid][aid][3][yid][xid])/2)/(restored_box[lid][aid][3][yid][xid]-restored_box[lid][aid][4][yid][xid])
                                local tw = math.log((xmax_-xmin_)/(restored_box[lid][aid][1][yid][xid]-restored_box[lid][aid][2][yid][xid]))
                                local th = math.log((ymax_-ymin_)/(restored_box[lid][aid][3][yid][xid]-restored_box[lid][aid][4][yid][xid]))
                                
                                if IoU[aid][yid][xid] > max_pos_iou then
                                    pos_best_match = {lid,aid,yid,xid,label,tx,ty,tw,th}
                                    max_pos_iou = IoU[aid][yid][xid]
                                end
                                
                                --assign boxes whose IoU > 0.5
                                IoU = torch.reshape(IoU,ar_num*fmSz[lid]*fmSz[lid])
                                local IoU_comp = torch.lt(pos_candidate_iou[{{startIdx,startIdx+ar_num*fmSz[lid]*fmSz[lid]-1}}],IoU) 
                                
                                pos_candidate_iou[{{startIdx,startIdx+ar_num*fmSz[lid]*fmSz[lid]-1}}][IoU_comp] = IoU[IoU_comp]
                                pos_candidate_label[{{startIdx,startIdx+ar_num*fmSz[lid]*fmSz[lid]-1}}][IoU_comp] = label
                                
                                pos_candidate_xmax[{{startIdx,startIdx+ar_num*fmSz[lid]*fmSz[lid]-1}}][IoU_comp] = xmax_
                                pos_candidate_xmin[{{startIdx,startIdx+ar_num*fmSz[lid]*fmSz[lid]-1}}][IoU_comp] = xmin_
                                pos_candidate_ymax[{{startIdx,startIdx+ar_num*fmSz[lid]*fmSz[lid]-1}}][IoU_comp] = ymax_
                                pos_candidate_ymin[{{startIdx,startIdx+ar_num*fmSz[lid]*fmSz[lid]-1}}][IoU_comp] = ymin_

                                startIdx = startIdx + ar_num*fmSz[lid]*fmSz[lid]

                            end

                            --pos assign(best match)
                            local lid = pos_best_match[1]
                            local aid = pos_best_match[2]
                            local yid = pos_best_match[3]
                            local xid = pos_best_match[4]
                            local label = pos_best_match[5]

                            table.insert(pos_set,pos_best_match)
                            table.insert(final_pos_conf_output,{lid,bid,aid,yid,xid})
                            table.insert(final_pos_conf_target,label)

                            
                            --[===[
                            --for debug
                            local img = inputs[bid]
                            local xmax = restored_box[lid][aid][1][yid][xid]
                            local xmin = restored_box[lid][aid][2][yid][xid]
                            local ymax = restored_box[lid][aid][3][yid][xid]
                            local ymin = restored_box[lid][aid][4][yid][xid]
                            img = drawRectangle(img,xmin,ymin,xmax,ymax,"r")
                            image.save(tostring(bid) .. "_" .. tostring(gid) ..  ".jpg",img)
                            --]===]

                        end

                        --pos assign(overlap > 0.5)
                        for pid = 1,table.getn(pos_set) do
                            local lid = pos_set[pid][1]
                            local aid = pos_set[pid][2]
                            local yid = pos_set[pid][3]
                            local xid = pos_set[pid][4]

                            local idx = combine_idx(lid,aid,yid,xid)
                            pos_candidate_iou[idx] = -1
                        end
                        pos_candidate_mask = pos_candidate_iou:ge(0.5)

                        pos_label = pos_candidate_label[pos_candidate_mask]
                        pos_xmax = pos_candidate_xmax[pos_candidate_mask]
                        pos_xmin = pos_candidate_xmin[pos_candidate_mask]
                        pos_ymax = pos_candidate_ymax[pos_candidate_mask]
                        pos_ymin = pos_candidate_ymin[pos_candidate_mask]
                        pos_idx = idx_tensor[pos_candidate_mask]

                        for pid = 1,torch.sum(pos_candidate_mask) do
                            
                            local lid,aid,yid,xid
                            local idx = pos_idx[pid]
                            lid,aid,yid,xid = parse_idx(idx)
                            
                            local label = pos_label[pid]

                            local xmax = pos_xmax[pid]
                            local xmin = pos_xmin[pid]
                            local ymax = pos_ymax[pid]
                            local ymin = pos_ymin[pid]

                            local tx = ((xmin+xmax)/2 - (restored_box[lid][aid][2][yid][xid]+restored_box[lid][aid][1][yid][xid])/2)/(restored_box[lid][aid][1][yid][xid]-restored_box[lid][aid][2][yid][xid])
                            local ty = ((ymin+ymax)/2 - (restored_box[lid][aid][4][yid][xid]+restored_box[lid][aid][3][yid][xid])/2)/(restored_box[lid][aid][3][yid][xid]-restored_box[lid][aid][4][yid][xid])
                            local tw = math.log((xmax-xmin)/(restored_box[lid][aid][1][yid][xid]-restored_box[lid][aid][2][yid][xid]))
                            local th = math.log((ymax-ymin)/(restored_box[lid][aid][3][yid][xid]-restored_box[lid][aid][4][yid][xid]))

                            table.insert(final_pos_conf_output,{lid,bid,aid,yid,xid})
                            table.insert(final_pos_conf_target,label)
                            table.insert(pos_set,{lid,aid,yid,xid,label})

                            if lid == 1 then
                                ar_num = 4
                            elseif lid < m then
                                ar_num = 6
                            else
                                ar_num = 5
                            end
                            table.insert(final_loc_output,{lid,bid,aid,yid,xid})
                            table.insert(final_loc_target,{tx,ty,tw,th})

                           
                        end
                        
                        if tot_iter % 100 == 0 then
                            --for debug
                            pos_img = inputs[bid]:clone()
                            for pid = 1,table.getn(pos_set) do
                                local lid = pos_set[pid][1]
                                local aid = pos_set[pid][2]
                                local yid = pos_set[pid][3]
                                local xid = pos_set[pid][4]
                                
                                local label = pos_set[pid][5]
                                label = classList[label]

                                local xmax = restored_box[lid][aid][1][yid][xid]
                                local xmin = restored_box[lid][aid][2][yid][xid]
                                local ymax = restored_box[lid][aid][3][yid][xid]
                                local ymin = restored_box[lid][aid][4][yid][xid]

                                --image.save(label .. tostring(pid) .. ".jpg",pos_img[{{},{math.max(ymin,1),math.min(ymax,imgSz)},{math.max(xmin,1),math.min(xmax,imgSz)}}])
                                pos_img = drawRectangle(pos_img,xmin,ymin,xmax,ymax,"g")
                            end
                            image.save("pos_" .. tostring(bid) .. ".jpg",pos_img)
                        end

                        --hard neg mining
                        startIdx = 1
                        neg_mask = torch.cmul(pos_candidate_iou:gt(0),pos_candidate_iou:lt(neg_thr)):eq(0) -- 1: iou <= 0 or iou >= 0.5
                        for lid = 1,m do
                            
                            if lid == 1 then
                                ar_num = 4
                            elseif lid < m then
                                ar_num = 6
                            else
                                ar_num = 5
                            end

                            for aid = 1,ar_num do

                                local neg_input = outputs[lid][{{bid},{(aid-1)*classNum+1,aid*classNum},{},{}}]
                                neg_input = SpatialSM:forward(neg_input):clone()
                                neg_input = neg_input[{{},{1,classNum-1},{},{}}]
                                neg_input,dummy = torch.max(neg_input,2)
                                neg_input[1][1][torch.reshape(neg_mask[{{startIdx,startIdx+fmSz[lid]*fmSz[lid]-1}}],fmSz[lid],fmSz[lid])] = -1
                                neg_input = torch.reshape(neg_input,1*1*fmSz[lid]*fmSz[lid])
                                neg_candidate_loss[{{startIdx,startIdx+fmSz[lid]*fmSz[lid]-1}}] = neg_input:type('torch.FloatTensor')
                                startIdx = startIdx + fmSz[lid]*fmSz[lid]

                            end

                        end
                        
                        neg_candidate_loss_cnt = torch.sum(neg_candidate_loss:ne(-1))
                        if neg_candidate_loss_cnt > pos_neg_ratio*table.getn(pos_set) then
                            neg_topk_val, neg_topk_idx = neg_candidate_loss:topk(pos_neg_ratio*table.getn(pos_set),true)
                            for nid = 1,neg_topk_idx:size()[1] do
                                local idx = neg_topk_idx[nid]
                                lid,aid,yid,xid = parse_idx(idx)

                                table.insert(final_neg_conf_output,{lid,bid,aid,yid,xid})
                                table.insert(final_neg_conf_target,negId)
                                table.insert(neg_set,{lid,aid,yid,xid,negId})
                            end
                        else
                            neg_valid_idx = idx_tensor[neg_candidate_loss:ne(-1)]
                            for nid = 1,neg_candidate_loss_cnt do
                                local idx = neg_valid_idx[nid]
                                lid,aid,yid,xid = parse_idx(idx)
                                
                                table.insert(final_neg_conf_output,{lid,bid,aid,yid,xid})
                                table.insert(final_neg_conf_target,negId)
                                table.insert(neg_set,{lid,aid,yid,xid,negId})
                            end

                        end

                        
                        
                        if tot_iter % 100 == 0 then
                            --for debug
                            neg_img = inputs[bid]:clone()
                            for nid = 1,table.getn(neg_set) do
                                local lid = neg_set[nid][1]
                                local aid = neg_set[nid][2]
                                local yid = neg_set[nid][3]
                                local xid = neg_set[nid][4]
                                
                                local xmax = restored_box[lid][aid][1][yid][xid]
                                local xmin = restored_box[lid][aid][2][yid][xid]
                                local ymax = restored_box[lid][aid][3][yid][xid]
                                local ymin = restored_box[lid][aid][4][yid][xid]
                                neg_img = drawRectangle(neg_img,xmin,ymin,xmax,ymax,"r")
                            end
                            image.save("neg" .. tostring(bid) .. ".jpg",neg_img)
                        end

                        
                    end

                    pos_conf_out = torch.Tensor(table.getn(final_pos_conf_output),classNum):type('torch.CudaTensor')
                    pos_conf_target = torch.Tensor(table.getn(final_pos_conf_target),1):type('torch.CudaTensor')
                    neg_conf_out = torch.Tensor(table.getn(final_neg_conf_output),classNum):type('torch.CudaTensor')
                    neg_conf_target = torch.Tensor(table.getn(final_neg_conf_target),1):type('torch.CudaTensor')
                    loc_out = torch.Tensor(table.getn(final_loc_output),4):type('torch.CudaTensor')
                    loc_target = torch.Tensor(table.getn(final_loc_target),4):type('torch.CudaTensor')

                    for cid = 1,table.getn(final_pos_conf_output) do
                        local lid = final_pos_conf_output[cid][1]
                        local bid = final_pos_conf_output[cid][2]
                        local aid = final_pos_conf_output[cid][3]
                        local yid = final_pos_conf_output[cid][4]
                        local xid = final_pos_conf_output[cid][5]

                        local feature = outputs[lid][{{bid},{(aid-1)*classNum+1,aid*classNum},{yid},{xid}}]

                        pos_conf_out[cid] = feature
                        pos_conf_target[cid] = final_pos_conf_target[cid]

                        pos_confusion_target = torch.Tensor(classNum-1):zero()
                        pos_confusion_target[pos_conf_target[cid][1]] = 1
                        pos_confusion:add(SM:forward(pos_conf_out[cid][{{1,classNum-1}}]:type('torch.CudaTensor')):clone(),pos_confusion_target)
                    end

                    for cid = 1,table.getn(final_neg_conf_output) do
                        local lid = final_neg_conf_output[cid][1]
                        local bid = final_neg_conf_output[cid][2]
                        local aid = final_neg_conf_output[cid][3]
                        local yid = final_neg_conf_output[cid][4]
                        local xid = final_neg_conf_output[cid][5]

                        local feature = outputs[lid][{{bid},{(aid-1)*classNum+1,aid*classNum},{yid},{xid}}]

                        neg_conf_out[cid] = feature
                        neg_conf_target[cid] = final_neg_conf_target[cid]

                        neg_confusion_target = torch.Tensor(classNum):zero()
                        neg_confusion_target[neg_conf_target[cid][1]] = 1
                        neg_confusion:add(SM:forward(neg_conf_out[cid]:type('torch.CudaTensor')):clone(),neg_confusion_target)
                    end


                    for cid = 1,table.getn(final_loc_output) do
                        
                        local lid = final_loc_output[cid][1]
                        local bid = final_loc_output[cid][2]
                        local aid = final_loc_output[cid][3]
                        local yid = final_loc_output[cid][4]
                        local xid = final_loc_output[cid][5]

                        if lid == 1 then
                            ar_num = 4
                        elseif lid < m then
                            ar_num = 6
                        else
                            ar_num = 5
                        end


                        local feature = outputs[lid][{{bid},{ar_num*classNum + (aid-1)*4+1, ar_num*classNum + (aid-1)*4+4},{yid},{xid}}]


                        loc_out[cid] = feature
                        loc_target[cid] = torch.Tensor(final_loc_target[cid])
                    end
                    
                    pos_class_error = crossEntropy:forward(pos_conf_out,pos_conf_target)
                    pos_class_dfdo = crossEntropy:backward(pos_conf_out,pos_conf_target):clone()
                    
                    neg_class_error = crossEntropy:forward(neg_conf_out,neg_conf_target)
                    neg_class_dfdo = crossEntropy:backward(neg_conf_out,neg_conf_target):clone()

                    loc_error = smoothL1:forward(loc_out,loc_target)/table.getn(final_loc_output)
                    loc_dfdo = smoothL1:backward(loc_out,loc_target)/table.getn(final_loc_output)

                    for cid = 1,table.getn(final_pos_conf_output) do
                        local lid = final_pos_conf_output[cid][1]
                        local bid = final_pos_conf_output[cid][2]
                        local aid = final_pos_conf_output[cid][3]
                        local yid = final_pos_conf_output[cid][4]
                        local xid = final_pos_conf_output[cid][5]
                                
                        --[===[
                        local xmax = restored_box[lid][aid][1][yid][xid]
                        local xmin = restored_box[lid][aid][2][yid][xid]
                        local ymax = restored_box[lid][aid][3][yid][xid]
                        local ymin = restored_box[lid][aid][4][yid][xid]
                        local label = final_pos_conf_target[cid][1]
                        local img = inputs[bid]
                        image.save(classList[label] .. tostring(cid) .. ".jpg",img[{{},{math.max(ymin,1),math.min(ymax,imgSz)},{math.max(xmin,1),math.min(xmax,imgSz)}}])
                        --]===]
   
                        tot_dfdo[lid][{{bid},{(aid-1)*classNum+1,aid*classNum},{yid},{xid}}] = tot_dfdo[lid][{{bid},{(aid-1)*classNum+1,aid*classNum},{yid},{xid}}] + pos_class_dfdo[cid]
                    end

                    for cid = 1,table.getn(final_neg_conf_output) do
                        local lid = final_neg_conf_output[cid][1]
                        local bid = final_neg_conf_output[cid][2]
                        local aid = final_neg_conf_output[cid][3]
                        local yid = final_neg_conf_output[cid][4]
                        local xid = final_neg_conf_output[cid][5]

                        tot_dfdo[lid][{{bid},{(aid-1)*classNum+1,aid*classNum},{yid},{xid}}] = tot_dfdo[lid][{{bid},{(aid-1)*classNum+1,aid*classNum},{yid},{xid}}] + neg_class_dfdo[cid]
                    end


                    for cid = 1,table.getn(final_loc_output) do
                        local lid = final_loc_output[cid][1]
                        local bid = final_loc_output[cid][2]
                        local aid = final_loc_output[cid][3]
                        local yid = final_loc_output[cid][4]
                        local xid = final_loc_output[cid][5]
                        
                       if lid == 1 then
                           ar_num = 4
                        elseif lid < m then
                            ar_num = 6
                        else
                            ar_num = 5
                        end

                        tot_dfdo[lid][{{bid},{ar_num*classNum+(aid-1)*4+1,ar_num*classNum+(aid-1)*4+4},{yid},{xid}}] = tot_dfdo[lid][{{bid},{ar_num*classNum+(aid-1)*4+1,ar_num*classNum+(aid-1)*4+4},{yid},{xid}}] + loc_dfdo[cid]
                    end

                    model:backward(inputs,tot_dfdo)

                    gradParams:div(curBatchDim)
                    class_error = pos_class_error + neg_class_error
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

            print(pos_confusion)
            print(neg_confusion)
            pos_confusion:zero()
            neg_confusion:zero()
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



