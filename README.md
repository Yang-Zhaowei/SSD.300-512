# Machine Learning Project 
## Task II Unbalanced samples

Detect and distinguish different types of mobile power by SSD(300/512) which based Resnet50.

Loss: Iou & Cross Entrop
<!-- 
#### 7 layers:
| AP |TrainSet|ValSet|
|:--:|:----------------:| :--------------: |
|core    |0.9511|0.8569|0.8857|
|coreless|0.9476|0.5838|0.9438|
|map     |0.9494|0.7203|0.9147| -->

####  Experment 1:
| AP |B:C:CL=1:1:1|B:C:CL=1:5:1|B:C:CL=1:6:1.2|7 layers|
|:------:|:----:|:----:|:----:|:---:|
|core    |0.7967|**0.8257**|0.821 |0.777|
|coreless|**0.8748**|0.8683|0.864 |0.877|
|map     |0.8358|**0.8470**|0.843 |0.827|

#### Experment 2:
|AP|512/1:1:1|512/1:5:1|512/1:6:1|300/1:1:1|300/1:5:1|
|:------:|:----:|:----:|:----:|:----:|:----:|
|core    |0.8150|**0.8507**|0.8295|0.6999|0.6883|
|coreless|0.9355|**0.9361**|0.9240|0.9212|0.9031|
|map     |0.8753|**0.8934**|0.8768|0.8106|0.7957|

The trained model weights file ```ssd120.pth```：

http://localhost

Group members: 王德辉、高思奇、黄振远、焦守坤、杨兆维

YangZhaowei

<!-- 151
ssd994:
AP for 带电芯充电宝 = 0.7238
AP for 不带电芯充电宝 = 0.8922
Mean AP = 0.8080
~~~~~~~~
Results:
0.724
0.892
0.808
~~~~~~~~ -->