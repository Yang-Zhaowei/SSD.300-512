# from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
from cv2 import cv2
import numpy as np

PB_CLASSES = ['带电芯充电宝', '不带电芯充电宝']  # always index 2
PB_ROOT = ""


class PBAnnotationTransform(object):
    """
    将充电宝的标注文件转换为bbox坐标和标签序号

    class_to_ind (dict,optional):dictionary lookup of classname -> index
    height
    width
    """

    def __init__(self, class_to_ind=None, test=False):
        self.class_to_ind = class_to_ind or dict(
            zip(PB_CLASSES, range(len(PB_CLASSES))))
        self.test = test

    def __call__(self, target, width, height, test):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for line in open(target, encoding='utf-8'):
            # difficult = int(obj.find('difficult').text) == 1
            #if not self.keep_difficult and difficult:
                #continue
            # name = obj.find('name').text.lower().strip()
            # pts = ['xmin', 'ymin', 'xmax', 'ymax']
            line = line.split()
            # print(line)
            if 'TIFF' in line[0]:
                line = line[1:]
            name = line[0]
            bndbox = []
            for i, pt in enumerate(line[1:]):
                pt = int(pt)
                cur_pt = pt/width if i % 2 == 0 else pt / height
                bndbox.append(cur_pt)
            if self.class_to_ind.get(name, -1) == -1:
                # label_idx=self.class_to_ind['other']
                continue
            else:
                label_idx = self.class_to_ind[name]

            bndbox.append(label_idx)
            if label_idx <= 1:
                res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class PBDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, image_path='data/Powerbank',  # image_sest=['coreless_5000', 'core_500'],
                 anno_path='data/Annotation', test=False, ratio=0.7,
                 transform=None, target_transform=PBAnnotationTransform(),):
        self.name = 'PowerBank'
        # self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.test = test
        # self.ids = list()
        # self.annos=list()
        self.image_path = image_path
        self.anno_path = anno_path
        self.ids = [osp.join(image_path, img)
                    for img in os.listdir(image_path)]
        self.annos = [osp.join(anno_path, anno)
                      for anno in os.listdir(anno_path)]
        if not test:
            self.ids = self.ids[:int(ratio*len(self.ids))]
            self.annos = self.annos[:int(ratio*len(self.annos))]
        else:
            self.ids = self.ids[int(ratio*len(self.ids)):]
            self.annos = self.annos[int(ratio*len(self.annos)):]
            with open('sub_test_core_coreless.txt', encoding='utf-8', mode='w+') as obj:
                for i in self.ids:
                    line = i.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                    obj.write(line+'\n')
        # for name in image_sets:
        #     self.rootpath = osp.join(self.root,name)
        #     ipath=osp.join(self.rootpath,'Image')
        #     imgs=[img for img in os.listdir(ipath)]

        #     num=len(imgs)
        #     if self.test:
        #         self.ids.extend(osp.join(ipath,img) for img in imgs[int(num*ratio):])

        #     else:
        #         self.ids.extend(osp.join(ipath,img) for img in imgs[:int(num*ratio)])

        # with open("sub_test_core_coreless.txt2",'w+',encoding='utf-8') as obj:
        #     for kkk in self.ids:
        #         obj.write("{}\n".format(kkk.split('/')[-1].split('.')[0]))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = img_id.split('.')[0].replace(
            self.image_path, self.anno_path)+'.txt'
        img = cv2.imread(img_id)
        # target = etree.parse(annopath).getroot()
        # target=[]
        # for line in open(annopath,encoding='utf-8'):
        #     line=line.split()
        #     if 'TIFF' in line[0]:
        #         line=line[1:]
        #     target.append(line)
        height, width, channels = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height, self.test)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(
                img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        target = img_id.split('.')[0].replace(
            self.image_path, self.anno_path)+'.txt'
        # anno = etree.parse(osp.join(self.rootpath, 'Annotations' ,img_id[:-4]) + '.xml').getroot()
        gt = self.target_transform(target, 1, 1, self.test)
        return img_id, gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


# with open("/home/yangzw/Pytorch/VOC/sub_test_core_coreless.txt",'w+',encoding='utf-8') as obj:
#     dataset=PBDetection(test=True)
#     print(1)
        # obj.write(i)
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


if __name__ == '__main__':
    datas = PBDetection(image_path='/home/yangzw/Pytorch/data/coreless_3000/Image/',
                        anno_path='/home/yangzw/Pytorch/data/coreless_3000/Annotation/')
    print(len(datas))
