from torchvision import models
import torch.nn as nn

from .utils import ConvModule


class Basenet(nn.Module):
    def __init__(self, model_name, feature_map):
        super(Basenet, self).__init__()
        self.normalize = {'type': 'BN'}
        lay = self.get_pretrainedmodel(model_name)
        self.model = self.add_extras(lay, len(feature_map))
        self.model_length = len(self.model)
        self.feature_map = feature_map

    def get_pretrainedmodel(self, pretrained=True):
        model = models.resnet50(pretrained=pretrained)
        #get the model lay,it's a list
        lay = nn.Sequential(*list(model.children())[:-2])
        # if model_name == 'resnet50':
        #     out_channels = 2048
        return lay  # ,out_channels

    def add_extras(self, lay, length=6):
        #在basenet上附加的6层卷积
        exts1 = nn.Sequential(
            ConvModule(2048, 256, 1, normalize=None, stride=1,
                       bias=True, inplace=False),
            ConvModule(256, 512, 3, normalize=None, stride=2, padding=1,
                       bias=True, inplace=False)
        )
        lay.add_module("exts1", exts1)

        # ConvModule是指定参数的卷积层、正则化和激活函数的集合，并且可以确定它们的顺序。
        exts2 = nn.Sequential(
            ConvModule(512, 128, 1, normalize=None, stride=1,
                       bias=True, inplace=False),
            ConvModule(128, 256, 3, normalize=None, stride=2, padding=1,
                       bias=True, inplace=False)

        )
        lay.add_module("exts2", exts2)

        exts3 = nn.Sequential(
            ConvModule(256, 128, 1, normalize=None, stride=1,
                       bias=True, inplace=False),
            ConvModule(128, 256, 3, normalize=None, stride=1, padding=0,
                       bias=True, inplace=False)
        )
        lay.add_module("exts3", exts3)

        if(length > 6):
            exts4 = nn.Sequential(
                ConvModule(256, 128, 1, normalize=None, stride=1,
                           bias=True, inplace=False),
                ConvModule(128, 256, 2, normalize=None, stride=1, padding=0,
                           bias=True, inplace=False)
            )
            lay.add_module("exts4", exts4)
        return lay

    def forward(self, x):
        outs = []

        for i in range(self.model_length):
            x = self.model[i](x)
            # print(x.shape)

            if i+1 in self.feature_map:

                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            # basenet 返回的是指定大小的feature map的元组
            return tuple(outs)


if __name__ == "__main__":
    import torch
    base = Basenet('resnet50', [6, 7, 8, 9, 10, 11, 12])
    a = torch.zeros(1, 3, 512, 512)
    out = base(a)
    for i in out:
        print(i.shape)
    # print(out)
