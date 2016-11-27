require 'torch'


function createBatch(t, shuffle, trainName, trainTarget)

            
    if t+batchSz-1 <= trainSz then
        inputs = torch.CudaTensor(batchSz,inputDim,imgSz,imgSz)
        targets = {}
        curBatchDim = batchSz
    else
        inputs = torch.CudaTensor(trainSz-t+1,inputDim,imgSz,imgSz)
        targets = {}
        curBatchDim = trainSz-t+1
    end

    label_count = torch.Tensor(classNum-1):zero()
    
    for i = t,math.min(t+batchSz-1,trainSz) do
        
        local input_name = trainName[shuffle[i]]
        local target = trainTarget[shuffle[i]]
        
        isTooMany = true
        while isTooMany == true do
            isTooMany = false
            for gid = 1,table.getn(target) do
                
                local label = target[gid][1]
                if label_count[label] > 3 then
                    isTooMany = true
                end
            end
            
            if isTooMany == true then
                new_idx = math.random(1,trainSz-(i-t+1))
                if new_idx >= (t-1)*batchSz+1 then
                    new_idx = new_idx + (i-t+1)
                end
                input_name = trainName[new_idx]
                target = trainTarget[new_idx]
            end
        end


        for gid = 1,table.getn(target) do
            label = target[gid][1]
            label_count[label] = label_count[label] + 1
        end


        local input = image.load(input_name)
        input = image.scale(input,imgSz,imgSz)
        r = input[{{1},{},{}}]:clone()
        b = input[{{3},{},{}}]:clone()
        
        input[{{1},{},{}}] = b
        input[{{3},{},{}}] = r

        inputs[i-t+1] = input
        table.insert(targets,target)
    end

    return inputs, targets, curBatchDim

end
     
