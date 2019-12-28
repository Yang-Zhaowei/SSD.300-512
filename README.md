# Machine Learning Project 
## Task II Unbalanced samples

Detect and distinguish different types of mobile power by SSD which based Resnet50.

Loss: GIOU & Cross Entrop

#### Epoch 100:
| AP |core/coreless_5500|core_3000|coreless_3000|
|:--:|:----------------:| :-----: | :---------: |
|core|0.861|0.836|0.722|
|coreless|0.896|0.566|0.895|
|map|0.879|0.701|0.834|


#### Epoch 300:
| AP |core/coreless_5500|core_3000|coreless_3000|
|:--:|:----------------:| :-----: | :---------: |
|core|0.872|0.810|0.778|
|coreless|0.904|0.588|0.904|
|map|0.888|0.699|0.841|

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