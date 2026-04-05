
import torch.nn as nn
# from utils.engine import *
import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import ModelEma, add_weight_decay
# from src.loss_functions.losses import AsymmetricLoss,CosLoss
from src.loss_functions.losses import CosLoss
# from randaugment import RandAugment
# from src.models.build_model_Decoder_2proto import MyModel
# from src.models.build_model_TF import MyModel
from models.clip_vit import CLIPVIT as CLIPVITBase
from models.prompt_model import CLIPVIT as CLIPVITDual
import torch.nn as nn
from utils.dual_LT_engine_grouplr import *
# from utils.LT_engine_coop import *
# from utils.engine_grouplr_2proto import *
# from utils.engine_nocos import MultiLabelEngine
# from src.data_loader.coco import COCO2014
from src.data_loader.datasets import build_dataset
from src.loss_functions.dbl import *
from src.loss_functions.asl import *

# from exp import *

import clip


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', help='path to image root (COCO: dir containing train2017/; VOC: dir containing VOCdevkit/)', default=None)
parser.add_argument('--dataset', default='coco-lt', type=str, choices=['voc-lt', 'coco-lt'], help='dataset name')
parser.add_argument('--clip-path', help='path to CLIP checkpoint (e.g. ViT-B-16.pt)', default=None)
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size')
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--seed', default='0', type=int, help='seed')
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--threshold', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--ctx_init', default='a photo of a', type=str, help='init context prompt')
parser.add_argument('--n_ctx', default=4, type=int, help='length M of context prompt when initializing')
parser.add_argument('--class_token_position', default='end', type=str, help='position of class token')
# parser.add_argument('--csc', default='', type=str, help='position of class token')

parser.add_argument('--loss_function', default='dbl', type=str, choices=['asl', 'bce', 'dbl', 'mls', 'FL', 'CBloss', 'R-BCE-Focal','NTR-Focal', 'DBloss-noFocal', 'CBloss-ntr', 'DBloss'], help='loss function')
parser.add_argument('--pretrain_clip', default='ViT16', type=str, choices=['RN50', 'ViT16'], help='pretrained clip backbone')
parser.add_argument('--topk',type = int,default=16)
parser.add_argument('--alpha',type = float,default=0.5)
parser.add_argument('--model-type', type=str, default='clip_vit_dual',
                    choices=['clip_vit', 'clip_vit_dual'],
                    help='对比实验: clip_vit=ViT+模板, clip_vit_dual=双视图+可学习prompt')

# 创新开关：1=动态文本提示词, 2=文本特征增强, 3=双分支局部与全局一致性损失
parser.add_argument('--use_dynamic_prompt', type=int, default=0, choices=[0, 1], help='1=启用动态文本提示词(根据图像特征调制context)')
parser.add_argument('--use_text_enhance', type=int, default=0, choices=[0, 1], help='1=启用文本特征增强')
parser.add_argument('--use_dual_consistency', type=int, default=0, choices=[0, 1], help='1=启用双分支局部与全局一致性损失')
parser.add_argument('--dual_consistency_weight', type=float, default=0.1, help='双分支一致性损失权重')
def main_coco():
    args = parser.parse_args()

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)

    # model_name = 'lr_asl_clip_vit_coco'
    model_name = 'Second_Stage_seed{}_al{}_lr{}_{}_clip_vit_{}_'.format(args.seed,args.alpha,args.lr,args.loss_function,args.dataset)
    # model_name = 'coop_seed{}_lr{}_{}_clip_vit_{}_'.format(args.seed,args.lr,args.loss_function,args.dataset)

    # 未指定 --clip-path 时使用模型名，clip.load 会自动下载到 ~/.cache/clip
    if args.clip_path and os.path.isfile(args.clip_path):
        pretrain_clip_path = args.clip_path
    elif args.pretrain_clip == "RN50":
        pretrain_clip_path = "RN50"  # 自动下载
    else:
        pretrain_clip_path = "ViT-B/16"  # 自动下载
    if args.data is None:
        args.data = '/data/yanjiexuan/coco' if args.dataset == 'coco-lt' else '/data2/yanjiexuan/voc'
    data_root = os.path.join(args.data, 'data') if args.dataset == 'coco-lt' and not os.path.isdir(os.path.join(args.data, 'train2017')) else args.data
    anno_dir = os.path.join(DATA_DIR, 'coco' if args.dataset == 'coco-lt' else 'voc')
    inp_name = os.path.join(anno_dir, 'coco_glove_300_coco_sequence.pkl' if args.dataset == 'coco-lt' else 'voc_glove_word2vec.pkl')
    freq_file = os.path.join(anno_dir, 'class_freq.pkl')

    print(f"Loading CLIP (backbone: {args.pretrain_clip})")
    clip_model, preprocess = clip.load(pretrain_clip_path, device='cpu', jit=False) # Must set jit=False for training
    # clip_model, preprocess = clip.load("ViT-B/16", device='cpu', jit=False) 


    # def convert_models_to_fp32(model): 
    #     for p in model.parameters(): 
    #         p.data = p.data.float() 
    #         p.grad.data = p.grad.data.float() 
    
    # clip.model.convert_weights(clip_model) # Actually this line is unnecessary since clip by default already on float16

    # # Build Model
    # clip_model, _ = clip.load(pretrain_clip_path, device='cpu', jit=False)
    # model = CLIPVIT(args, clip_model)
    # convert_models_to_fp32(model)
    # model = model.to(args.device)

    # # Load CLIP
    # clip_model, _ = clip.load(args.clip_path, jit=False)
    # clip_model.eval()
    # clip_model = clip_model.to(args.device)
    # loading dataset
    # train_dataset = COCO2014(root=args.data,
    #                          phase='train',
    #                          transform=transforms.Compose([
    #                              transforms.Resize((args.image_size, args.image_size)),
    #                              CutoutPIL(cutout_factor=0.5),
    #                              RandAugment(),
    #                              transforms.ToTensor(),
    #                          ]),
    #                          inp_name='data/coco/coco_glove_word2vec_mlgcn.pkl'
    #                          )
    if args.dataset == 'coco-lt':
        dataset_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
            'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
            'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
            ]
        train_dataset = build_dataset(dataset=args.dataset, split='train', inp_name=inp_name, data_root=data_root, anno_dir=anno_dir)
        val_dataset = build_dataset(dataset=args.dataset, split='test', inp_name=inp_name, data_root=data_root, anno_dir=anno_dir)
    else:
        dataset_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor'
            ]
        train_dataset = build_dataset(dataset=args.dataset, split='train', inp_name=inp_name, data_root=data_root, anno_dir=anno_dir)
        val_dataset = build_dataset(dataset=args.dataset, split='test', inp_name=inp_name, data_root=data_root, anno_dir=anno_dir)

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # loss functions (freq_file already set above)
    if args.loss_function == 'bce':
        loss_function = nn.BCEWithLogitsLoss()
    if args.loss_function == 'mls':
        loss_function = nn.MultiLabelSoftMarginLoss()
    if args.loss_function == 'asl':
        loss_function = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    if args.loss_function == 'FL':
        if args.dataset == 'coco-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func=None,
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                loss_weight=1.0, freq_file=freq_file
            )
        elif args.dataset == 'voc-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func=None,
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
                loss_weight=1.0, freq_file=freq_file
            )
    if args.loss_function == 'CBloss': #CB
        if args.dataset == 'coco-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='CB',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                loss_weight=10.0, freq_file=freq_file
            )
        elif args.dataset == 'voc-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='CB',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                loss_weight=10.0, freq_file=freq_file
            )
    if args.loss_function == 'DBloss-noFocal': # DB-0FL
        if args.dataset == 'coco-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=False, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                loss_weight=0.5, freq_file=freq_file
            )
        elif args.dataset == 'voc-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=False, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
                loss_weight=0.5, freq_file=freq_file
            )
    if args.loss_function == 'dbl':
        if args.dataset == 'coco-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                loss_weight=1.0, freq_file=freq_file
            )
        elif args.dataset == 'voc-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
                loss_weight=1.0, freq_file=freq_file
            )


    # hyper-parameters
    args = {
        'num_classes': args.num_classes,
        'max_epoch': 120,
        'resume': args.resume,
        'evaluation': args.evaluate,
        'threshold': args.threshold,
        'dataset': args.dataset,
        'ctx_init': args.ctx_init,
        'n_ctx': args.n_ctx,
        'class_token_position': args.class_token_position,
        'lr': args.lr,
        'train': True,
        'topk': args.topk,
        'alpha': args.alpha,
        'model_type': args.model_type,
        'use_dynamic_prompt': args.use_dynamic_prompt,
        'use_text_enhance': args.use_text_enhance,
        'use_dual_consistency': args.use_dual_consistency,
        'dual_consistency_weight': args.dual_consistency_weight,
        'num_heatmaps': 10,
        'heatmap_save_dir': 'figures/heatmaps',
    }
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))
    print("model-type: {}".format(args['model_type']))
    
    # load model（对比实验: clip_vit / clip_vit_dual）
    ModelClass = CLIPVITDual if args['model_type'] == 'clip_vit_dual' else CLIPVITBase
    print('creating model {} ({})'.format(model_name, args['model_type']))
    models = nn.ModuleList([])
    if args['evaluation']:
        model = ModelClass(args, classnames=dataset_classes, clip_model=clip_model)
        models.append(model)
    else:
        regular_model = ModelClass(args, classnames=dataset_classes, clip_model=clip_model).cuda()
        # regular_model = torch.nn.DataParallel(regular_model)
        # if args['dataset']=='voc-lt': 
        #     checkpoint = torch.load(args['resume'])
        #     filtered_dict = {k.replace("module.",""): v for k, v in checkpoint['state_dict'].items()}
        #     regular_model.load_state_dict(filtered_dict)
        # elif args['dataset']=='coco-lt': 
        #     checkpoint = torch.load(args['resume'])
        #     filtered_dict = {k.replace("module.",""): v for k, v in checkpoint['state_dict'].items()}
        #     regular_model.load_state_dict(filtered_dict)
        ema_model = ModelEma(regular_model, 0.998)  # 0.9997^641=0.82
        models.append(regular_model)
        models.append(ema_model)

    # clip_vit_dual 仅训练 prompt_learner（含动态提示）及可选的 text_enhancer
    if hasattr(models[0], 'prompt_learner'):
        for name, param in models[0].named_parameters():
            if "prompt_learner" not in name and "text_enhancer" not in name:
                param.requires_grad = False




    # set optimizer
    Epochs = 80
    weight_decay = 1e-4
    # loss function
    criterion = nn.ModuleList([])
    # overall classification loss function
    
    # criterion.append(AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True))
    criterion.append(loss_function)
    criterion.append(CosLoss(args['num_classes'], 0.5))
    # criterion.append(AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True))
    # criterion.append(AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True))
    # visual space classification loss function
    # semantic space classification loss function

    # criterion.append(CosLoss(args['num_classes'],0.5))
    # for para in models[0].Backbone.parameters():
        # para.requires_grad = False

    # parameters = add_weight_decay(models[0].prompt_learner, weight_decay)
    # parameters = add_weight_decay(models[0], weight_decay)
    parameters = add_weight_decay(models, weight_decay)
    optimizer = torch.optim.AdamW(params = parameters, lr=args['lr'], weight_decay=0) # true wd, filter_bias_and_bn

    # print("Turning off gradients in the image encoder and text encoder")
    # for name, param in models[0].named_parameters():
    #     if "prompt_learner" not in name:
    #         param.requires_grad = False

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
    engine.learning(models, clip_model,train_loader, val_loader, criterion, optimizer, scheduler, model_name)



if __name__ == '__main__':
    main_coco()
