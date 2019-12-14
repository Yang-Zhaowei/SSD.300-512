import time
import argparse
import os
import sys
import json

from tqdm import tqdm
import torch.utils.data as data
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch

from config import pb
from utils import MultiBoxLoss
from data import *
from model_ssd import build_ssd

sys.path.append(os.getcwd())


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


'''
from eval import test_net
'''

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--image_path',default=None,type=str)
parser.add_argument('--anno_path',default=None,type=str)
# train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='PB',
                    type=str, help='Dataset, only powerbank')
parser.add_argument('--batch_size', default=36, type=int,
                    help='Batch size for training')
parser.add_argument('--max_epoch', default=50, type=int,
                    help='Max Epoch for training')
parser.add_argument('--resume', default="work_dir/ssd495.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--loss_weights', default=[1, 5, 1],
                     help='Weight of classes in loss function')
parser.add_argument('--lr', '--learning-rate', default=8e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--work_dir', default='work_dir/',
                    help='Directory for saving checkpoint models')
        
parser.add_argument('--weight', default=5, type=int)

args = parser.parse_args()

weight = args.weight

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.work_dir):
    os.mkdir(args.work_dir)


def train():
    '''
    get the dataset and dataloader
    '''
    if args.dataset == 'PB':
        if not os.path.exists(PB_ROOT):
            parser.error('Must specify dataset_root if specifying dataset')

        cfg = pb
        dataset = PBDetection(image_path=args.image_path,anno_path=args.anno_path,
                              transform=SSDAugmentation(cfg['min_dim'],
                                                        mean=cfg['mean'], std=cfg['std']))

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    #build, load,  the net
    ssd_net = build_ssd('train', size=cfg['min_dim'], cfg=cfg)

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_state_dict(torch.load(args.resume))

    if args.cuda:
        net = ssd_net.cuda()
    net.train()

    #optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    #loss
    # print(cfg['losstype'])
    criterion = MultiBoxLoss(cfg=cfg, overlap_thresh=0.5,
                             prior_for_matching=True, bkg_label=0,
                             neg_mining=True, neg_pos=3, neg_overlap=0.5,
                             encode_target=False, weight=args.loss_weights, use_gpu=args.cuda, loss_name=cfg['losstype'])

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name, epoch_size)
    iteration = args.start_iter
    step_index = 0
    loc_loss = 0
    conf_loss = 0
    save_folder = args.work_dir+cfg['work_name']
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    Lossc,Lossl,Loss=[],[],[]
    for epoch in range(args.max_epoch):
        for ii, batch_iterator in tqdm(enumerate(data_loader)):
            iteration += 1

            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            # load train data
            images, targets = batch_iterator
            #print(images,targets)
            if args.cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            else:
                images = images
                targets = [ann for ann in targets]
            t0 = time.time()
            out = net(images, 'train')
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = weight * loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            #print(iteration)
            if iteration % 10 == 0:
                print('timer: {:.4f} sec.' .format(t1 - t0))
                print('iter {} Loss: {:.4f} '.format(repr(iteration),loss.item()), end=' ')
                Loss.append(loss.item())
        
        Lossc.append(loc_loss)
        Lossl.append(conf_loss)
        if epoch % 10 == 0 and epoch > 60:  # epoch>1000 and epoch % 50 == 0:
            print('Saving state, iter:', iteration)
            #print('loss_l:'+weight * loss_l+', loss_c:'+'loss_c')

            torch.save(net.state_dict(), save_folder+'/ssd' +
                       repr(epoch)+'_.pth')

        loc_loss = 0
        conf_loss = 0
    torch.save(net.state_dict(), save_folder+'/ssd' +
               repr(epoch) + str(args.weight) + '_.pth')
    with open(save_folder+'/lossc.json','w+',encoding='utf-8') as obj:
        json.dump(Lossc,obj,ensure_ascii=False)
    with open(save_folder+'/lossl.json','w+',encoding='utf-8') as obj:
        json.dump(Lossl,obj,ensure_ascii=False)
    with open(save_folder+'/loss.json','w+',encoding='utf-8') as obj:
        json.dump(Loss,obj,ensure_ascii=False)


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(param_group['lr'])


if __name__ == '__main__':
    train()
