from xml.etree.ElementTree import parse
import os

db_dir = "/media/sda1/Data/PASCAL_VOC/VOCdevkit/VOC2007_trainval/Annotations/"

fileList = os.listdir(db_dir)
for file_name in fileList:
     
    if file_name == "parsed":
        continue

    filePath = db_dir + file_name
    tree = parse(filePath)
    root = tree.getroot()
    parsed = []

    for annot in root.iter("annotation"):


        for obj in annot.findall("object"):
            label = obj.findtext("name")
            
            for coord in obj.findall("bndbox"):
                x_max = coord.findtext("xmax")
                x_min = coord.findtext("xmin")
                y_max = coord.findtext("ymax")
                y_min = coord.findtext("ymin")

            parsed = parsed + [str(label) + ","  + str(x_max) + "," + str(x_min) + ","+ str(y_max) + "," + str(y_min)]
        
    fp = open(db_dir + "parsed/" + file_name[:-3] + "txt","w")
    for elem in parsed:
        print>>fp, elem
    fp.close()


