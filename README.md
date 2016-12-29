# SSD-Single-Shot-MultiBox-Detector
(SSD: Single Shot MultiBox Detector, Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, https://arxiv.org/abs/1512.02325)

##Start
th run_me.lua

##Implementation
Implemented the SSD300 baseline (fully implemented except data augmentation)

VGG weight is from https://github.com/weiliu89/caffe/tree/ssd

Applying bounding box regression make worse results (BUG)

##Qualitative results
![4](https://cloud.githubusercontent.com/assets/13601723/20825865/68c955dc-b8aa-11e6-95cd-abe3eba25ffd.jpg)
![28](https://cloud.githubusercontent.com/assets/13601723/20825867/6bd7dab4-b8aa-11e6-9044-cc7da83dc84c.jpg)
![1084](https://cloud.githubusercontent.com/assets/13601723/20825868/714e8178-b8aa-11e6-9cf3-e69a6d53184b.jpg)
![288](https://cloud.githubusercontent.com/assets/13601723/20825869/7172b5ca-b8aa-11e6-898c-c6bfe8b2f164.jpg)
![284](https://cloud.githubusercontent.com/assets/13601723/20825870/718941aa-b8aa-11e6-8527-f0940fcb1689.jpg)
![51](https://cloud.githubusercontent.com/assets/13601723/20825871/718b33ac-b8aa-11e6-9af1-e88074c9c5f8.jpg)
![37](https://cloud.githubusercontent.com/assets/13601723/20825872/718ecc92-b8aa-11e6-8d21-d5b7efe55453.jpg)



