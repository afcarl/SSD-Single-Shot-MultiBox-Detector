require "torch"
require "image"
require "math"
require "LuaXML"
dofile "etc.lua"



function label_to_num(label)

    if label == "aeroplane" then
        return 1
    elseif label == "bicycle" then
        return 2
    elseif label == "bird" then
        return 3
    elseif label == "boat" then
        return 4
    elseif label == "bottle" then
        return 5
    elseif label == "bus" then
        return 6
    elseif label == "car" then
        return 7
    elseif label == "cat" then
        return 8
    elseif label == "chair" then
        return 9
    elseif label == "cow" then
        return 10
    elseif label == "diningtable" then
        return 11
    elseif label == "dog" then
        return 12
    elseif label == "horse" then
        return 13
    elseif label == "motorbike" then
        return 14
    elseif label == "person" then
        return 15
    elseif label == "pottedplant" then
        return 16
    elseif label == "sheep" then
        return 17
    elseif label == "sofa" then
        return 18
    elseif label == "train" then
        return 19
    elseif label == "tvmonitor" then
        return 20
    end
    
end

function load_data(mode)

    target = {}
    name = {}

    if mode == "train" then
        print("training data loading...")
        dataNum = 3
    elseif mode == "test" then
        print("test data loading...")

        db_dir_ = db_dir .. "VOC2012_test/"
        imgDir = db_dir_ .. 'JPEGImages/'
        f = io.popen('ls ' .. imgDir)
        testFileList = {}
        for name in f:lines() do table.insert(testFileList,imgDir .. name) end

        return {}, testFileList
    end
    
    for did = 1,dataNum do

        if did == 1 then db_dir_ = db_dir .. "VOC2012_trainval/" end
        if did == 2 then db_dir_ = db_dir .. "VOC2007_trainval/" end
        if did == 3 then db_dir_ = db_dir .. "VOC2007_test/" end

        imgDir = db_dir_ .. 'JPEGImages/'
        annotDir = db_dir_ .. 'Annotations/parsed/'
        f = io.popen('ls ' .. annotDir)
        annotFileList = {}
        for name in f:lines() do table.insert(annotFileList,name) end
       
        for fid = 1,#annotFileList do
            
            --img load
            img = image.load(imgDir .. annotFileList[fid]:sub(1,-4) .. "jpg")
            local imgHeight = img:size()[2]
            local imgWidth = img:size()[3]
            --img = image.scale(img,imgSz,imgSz)

            --name save
            table.insert(name,imgDir .. annotFileList[fid]:sub(1,-4) .. "jpg")

            --label save
            target_per_sample = {}
            for line in io.lines(annotDir .. annotFileList[fid]) do
                
                label, xmax, xmin, ymax, ymin = line:match("([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)")
                
                label = label_to_num(label)
                xmax = tonumber(xmax)
                xmin = tonumber(xmin)
                ymax = tonumber(ymax)
                ymin = tonumber(ymin)
                
                --[===[
                --for debug
                img = drawRectangle(img,xmin,ymin,xmax,ymax)
                image.save(tostring(fid) .. ".jpg",img)
                --]===]
                
                table.insert(target_per_sample,{label,xmax,xmin,ymax,ymin,imgWidth,imgHeight})
            end
           
            table.insert(target,target_per_sample)
        end
    end

    return target, name
end

