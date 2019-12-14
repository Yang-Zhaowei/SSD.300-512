# config.py
import os.path
HOME = os.path.join(os.getcwd()) #../path
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

# SSD300 CONFIGS

pb= {
    'model':"resnet50",
    'losstype':'Giou',
    'num_classes':3,
    'mean':(127.5, 127.5, 127.5),
    'std':(1.0,1.0,1.0),
    'lr_steps': (80000, 100000,120000),
    'max_iter': 120000,
    'max_epoch':80,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'basenet_out':[512,1024,2048,512,256,256],
    'neck_out':[256,256,256,256,256,256],
    'steps':[8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'nms_kind': "diounms",
    'beta1':1.0,
    'name': 'PB',
    'work_name':"SSD300_PB",
}
