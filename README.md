# Machine Learning Project 
## Task II Unbalanced samples

Detect and distinguish different types of mobile power by SSD(300/512) which based Resnet50.

Loss: Iou & Cross Entrop

#### Experment:

|AP|512/1:1:1|512/1:5:1|512/1:6:1|300/1:1:1|300/1:5:1|300/1:6:1|
|:------:|:----:|:----:|:----:|:----:|:----:|:----:|
|core    |0.8150|**0.8507**|0.8295|0.6899|0.7191|0.7119|
|coreless|0.9355|**0.9361**|0.9240|0.9212|0.8965|0.8955|
|map     |0.8753|**0.8934**|0.8768|0.8056|0.8078|0.8037|

The trained model weights file ```ssd120.pth```：

http://localhost

Group members: 王德辉、高思奇、黄振远、焦守坤、杨兆维

YangZhaowei