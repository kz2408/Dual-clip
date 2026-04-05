import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CutoutPIL, ModelEma, add_weight_decay
from src.loss_functions.losses import AsymmetricLoss,CosLoss
from randaugment import RandAugment
import torch.nn as nn
from utils.engine_grouplr import *
# from utils.engine_interopt import *
# from utils.engine_grouplr_2proto import *
from src.data_loader.nus_fsl import NUSWIDEClassification_fsl
from src.data_loader.voc_fsl import Voc2007Classification_fsl
from src.data_loader.coco_fsl import CocoDataset

from models.prompt_model import CLIPVIT

import clip

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', help='path to dataset', default='/data/yanjiexuan/coco')
parser.add_argument('--dataset', help='dataset', default='coco')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--threshold', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--ctx_init', default='a photo of a', type=str, help='init context prompt')
parser.add_argument('--n_ctx', default=4, type=int, help='length M of context prompt when initializing')
parser.add_argument('--class_token_position', default='end', type=str, help='position of class token')

parser.add_argument('--pretrain_clip', default='ViT16', type=str, choices=['RN50', 'ViT16'], help='pretrained clip backbone')
parser.add_argument('--topk',type = int,default=16)
parser.add_argument('--lamda',type = float,default=1.0)
parser.add_argument('--alpha',type = float,default=0.5)

def main_coco():
    args = parser.parse_args()

    model_name = 'fsl_First_Stage_al{}_lr{}_clip_vit_topk{}_{}'.format(args.alpha,args.lr,args.topk,args.dataset)
        
    if args.dataset == 'coco':

        inp_seman = 'data/coco/coco_glove_300_coco_sequence.pkl'

        args.num_classes = 64
        args.data = '/data2/yanjiexuan/coco/data'
        dataset_classes = [
            'person', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'traffic_light', 'fire_hydrant',
            'parking_meter', 'bench', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'umbrella',
            'handbag', 'tie', 'suitcase', 'skis',
            'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 
            'tennis_racket', 'bottle', 'wine_glass',
            'knife', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'carrot', 'hot_dog', 'pizza', 'donut', 'cake',
            'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'cell_phone',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
            ]
        train_dataset = CocoDataset(root_dir=args.data,
                             set_name='train2014',
                             transform=transforms.Compose([
                                 transforms.Resize((args.image_size, args.image_size)),
                                 CutoutPIL(cutout_factor=0.5),
                                 RandAugment(),
                                 transforms.ToTensor(),
                             ]),
                             unseen_set=False,
                             return_ids=False,
                             inp_name=inp_seman
                             )

        val_dataset = CocoDataset(root_dir=args.data,
                             set_name='val2014',
                             transform=transforms.Compose([
                               transforms.Resize((args.image_size, args.image_size)),
                               transforms.ToTensor(),
                             ]),
                             unseen_set=False,
                             return_ids=False,
                             inp_name=inp_seman
                             )

    elif args.dataset == 'nus':
        args.num_classes = 65
        args.data = '/data2/yanjiexuan/nuswide'
        inp_seman = 'data/nuswide/nuswide_glove_word2vec.pkl'
        train_dataset = NUSWIDEClassification_fsl(args.data, './idx/nus/classification_Train_base.csv',
                                               word_emb_file=inp_seman,
                                               transform=transforms.Compose([
                                                   transforms.Resize((args.image_size, args.image_size)),
                                                   CutoutPIL(cutout_factor=0.5),
                                                   RandAugment(),
                                                   transforms.ToTensor(),
                                               ]))
        val_dataset = NUSWIDEClassification_fsl(args.data, './idx/nus/classification_Test_base.csv',
                                             word_emb_file=inp_seman,
                                             transform=transforms.Compose([
                                                 transforms.Resize((args.image_size, args.image_size)),
                                                 transforms.ToTensor()
                                             ]))
        
    elif args.dataset == 'voc':
        args.num_classes = 14
        dataset_classes = ['bottle','car','chair','diningtable','person','bicycle','motorbike','aeroplane','bird','boat','bus','cow','horse','train']
        args.data = '/data2/yanjiexuan/voc'
        inp_seman = 'data/voc/voc_glove_word2vec.pkl'
        train_dataset = Voc2007Classification_fsl(args.data,'/home/yanjiexuan/multi-label-fsl/RC-Tran/idx/voc/classification_Train_base.csv',
                                            word_emb_file = inp_seman,
                                            transform=transforms.Compose([
                                                   transforms.Resize((args.image_size, args.image_size)),
                                                   CutoutPIL(cutout_factor=0.5),
                                                   RandAugment(),
                                                   transforms.ToTensor(),
                                               ]))
        val_dataset = Voc2007Classification_fsl(args.data,'/home/yanjiexuan/multi-label-fsl/RC-Tran/idx/voc/classification_Test_base.csv',
                                           word_emb_file = inp_seman,
                                           transform=transforms.Compose([
                                                   transforms.Resize((args.image_size, args.image_size)),
                                                   transforms.ToTensor(),
                                               ]))
    else:
        raise NotImplementedError
    
    if args.pretrain_clip == "RN50":
        pretrain_clip_path = '/data2/yanjiexuan/huggingface/openai/pretrained/RN50.pt'
    elif args.pretrain_clip == "ViT16":
        pretrain_clip_path = '/data2/yanjiexuan/huggingface/openai/pretrained/ViT-B-16.pt'

    print(f"Loading CLIP (backbone: {args.pretrain_clip})")
    clip_model, preprocess = clip.load(pretrain_clip_path, device='cpu', jit=False) # Must set jit=False for training
    
    # loading dataset

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # hyper-parameters
    args = {
        'num_classes': args.num_classes,
        'max_epoch': 120,
        'resume': args.resume,
        'evaluation': args.evaluate,
        'threshold': args.threshold,
        'lr': args.lr,
        'dataset': args.dataset,
        'ctx_init': args.ctx_init,
        'n_ctx': args.n_ctx,
        'class_token_position': args.class_token_position,
        'lr': args.lr,
        'train': True,
        'topk': args.topk,
        'lamda': args.lamda,
        'alpha': args.alpha,
        "inp_seman":inp_seman
    }
    print(args)
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))
    
    # load model
    print('creating model {}'.format(model_name))
    models = nn.ModuleList([])
    if args['evaluation']:
        model = CLIPVIT(args)
        models.append(model)
    else:
        regular_model = CLIPVIT(args,classnames=dataset_classes, clip_model=clip_model).cuda()
        # regular_model = torch.nn.DataParallel(regular_model)
        ema_model = ModelEma(regular_model, 0.998)  # 0.9997^641=0.82
        models.append(regular_model)
        models.append(ema_model)

    for name, param in models[0].text_encoder.named_parameters():
        param.requires_grad = False
    
    for name, param in models[0].prompt_learner.named_parameters():
        param.requires_grad = False


    # set optimizer
    Epochs = 80
    weight_decay = 1e-4
    # loss function
    criterion = nn.ModuleList([])
    # overall classification loss function
    
    criterion.append(AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True))
    # criterion.append(CosLoss(args['num_classes'], 0.5))
    # criterion.append(AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True))
    # criterion.append(AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True))
    # visual space classification loss function
    # semantic space classification loss function

    # criterion.append(CosLoss(args['num_classes'],0.5))
    # for para in models[0].Backbone.parameters():
        # para.requires_grad = False

    parameters = add_weight_decay(models[0], weight_decay)
    optimizer = torch.optim.AdamW(params=parameters, lr=args['lr'], weight_decay=0) # true wd, filter_bias_and_bn

    # param_dicts = [{"params": [p for n, p in models[0].named_parameters() if p.requires_grad]}]
    # optimizer = torch.optim.AdamW(params=param_dicts, lr=args['lr'], weight_decay=weight_decay)  # true wd, filter_bias_and_bn

    # param_dicts = [{"params": [p for n, p in models[0].named_parameters() if p.requires_grad]},]
    # optimizer = getattr(torch.optim,'AdamW')(
    #         param_dicts,
    #         lr=args['lr'],
    #         betas=(0.9, 0.99), eps=1e-08, weight_decay=weight_decay
    #     )
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args['lr'], steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.1)

    engine = MultiLabelEngine(args)
    engine.learning(models, clip_model, train_loader, val_loader, criterion, optimizer, scheduler,model_name)


if __name__ == '__main__':
    main_coco()
