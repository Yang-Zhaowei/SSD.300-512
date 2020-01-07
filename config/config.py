# config.py
import os.path
HOME = os.path.join(os.getcwd())  # ../path
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

# SSD300 CONFIGS

pb300 = {
    'model': "resnet50",
    'losstype': 'Iou',
    'num_classes': 3,
    'mean': (127.5, 127.5, 127.5),
    'std': (1.0, 1.0, 1.0),
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'max_epoch': 120,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'basenet_lay': [6, 7, 8, 9, 10, 11],
    'basenet_out': [512, 1024, 2048, 512, 256, 256],
    'neck_out': [256, 256, 256, 256, 256, 256],
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'nms_kind': "nms",
    'beta1': 1.0,
    'name': 'PB',
    'work_name': "SSD300_PB",
}
# SSD512 CONFIGS

pb512 = {
    'model': "resnet50",
    'losstype': 'Iou',
    'num_classes': 3,
    'mean': (127.5, 127.5, 127.5),
    'std': (1.0, 1.0, 1.0),
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'max_epoch': 120,
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'basenet_lay': [6, 7, 8, 9, 10, 11, 12],
    'basenet_out': [512, 1024, 2048, 512, 256, 256, 256],
    'neck_out': [256, 256, 256, 256, 256, 256, 256],
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],
    'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'nms_kind': "nms",
    'beta1': 1.0,
    'name': 'PB',
    'work_name': "SSD512_PB",
}
