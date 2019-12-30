# Machine Learning Project 
## Task II Unbalanced samples

Detect and distinguish different types of mobile power by SSD which based Resnet50.

Loss: GIOU & Cross Entrop
<!-- 
#### 7 layers:
| AP |TrainSet|ValSet|
|:--:|:----------------:| :--------------: |
|core    |0.9511|0.8569|0.8857|
|coreless|0.9476|0.5838|0.9438|
|map     |0.9494|0.7203|0.9147| -->

####  Result:
| AP |B:C:CL=1:1:1|B:C:CL=1:5:1|B:C:CL=1:6:1.2|7 layers|
|:------:|:----:|:----:|:----:|:---:|
|core    |0.7967|0.8257|0.821 |0.777|
|coreless|0.8748|0.8683|0.864 |0.877|
|map     |0.8358|0.8470|0.843 |0.827|

The weights file of trained model and assignment reports would be updated later!

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