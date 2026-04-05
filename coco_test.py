
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
from models.clip import CustomCLIP as CLIPZeroShot
from models.coop import CustomCLIP as CoopCLIP
from models.clip_vit import CLIPVIT as CLIPVITBase
from models.prompt_model import CLIPVIT as CLIPVITDual
import torch.nn as nn
from utils.LT_engine_test import *
# from utils.LT_engine_grouplr_loss import *
# from utils.engine_grouplr_2proto import *
# from utils.engine_nocos import MultiLabelEngine
# from src.data_loader.coco import COCO2014
from src.data_loader.datasets import build_dataset
from src.loss_functions.dbl import *
from src.loss_functions.asl import *

from src.helper_functions.metrics import *

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
parser.add_argument('--resume', default='/data2/yanjiexuan/checkpoints/MKT/MKT_LT_checkpoint/ema_First_Stage_lr5e-05_dbl_clip_vit_voc-lt_best_91.587_e13.ckpt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', default= True, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--ctx_init', default='a photo of a', type=str, help='init context prompt')
parser.add_argument('--n_ctx', default=4, type=int, help='length M of context prompt when initializing')
parser.add_argument('--class_token_position', default='end', type=str, help='position of class token')
# parser.add_argument('--csc', default='', type=str, help='position of class token')

parser.add_argument('--loss_function', default='dbl', type=str, choices=['asl', 'bce', 'dbl', 'mls', 'FL', 'CBloss', 'R-BCE-Focal','NTR-Focal', 'DBloss-noFocal', 'CBloss-ntr', 'DBloss'], help='loss function')
parser.add_argument('--pretrain_clip', default='ViT16', type=str, choices=['RN50', 'ViT16'], help='pretrained clip backbone')
parser.add_argument('--topk',type = int,default=32)
parser.add_argument('--lamda',type = float,default=1.0)
parser.add_argument('--alpha',type = float,default=0.4)
parser.add_argument('--model-type', type=str, default='coop',
                    choices=['clip', 'coop', 'clip_vit', 'clip_vit_dual'],
                    help='对比实验: clip=零样本CLIP, coop=CoOp, clip_vit=ViT+模板, clip_vit_dual=双视图+可学习prompt')
parser.add_argument('--Mamba_en',type = int ,default=0)
parser.add_argument('--Fusion_en',type = int ,default=0)



def main_coco():
    args = parser.parse_args()

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)

    # model_name = 'lr_asl_clip_vit_coco'
    # model_name = 'First_Stage_seed{}_al{}_lr{}_{}_clip_vit_topk{}_{}_no_prompt'.format(args.seed,args.alpha,args.lr,args.loss_function,args.topk,args.dataset)
    # model_name = 'First_Stage_seed{}_al{}_lr{}_{}_clip_vit_topk{}_{}_only_prompt'.format(args.seed,args.alpha,args.lr,args.loss_function,args.topk,args.dataset)
    # model_name = 'First_Stage_seed{}_al{}_lr{}_{}_clip_vit_topk{}_{}'.format(args.seed,args.alpha,args.lr,args.loss_function,args.topk,args.dataset)

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

    freq_file = os.path.join(anno_dir, 'class_freq.pkl')
    # loss functions
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


    # hyper-parameters（注意此处 args 被覆盖为 dict，model_type 需一并传入）
    args = {
        'batch_size': args.batch_size,
        'workers': args.workers,
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
        'train': False,
        'topk': args.topk,
        'lamda': args.lamda,
        'alpha': args.alpha,
        'model_type': args.model_type,
        'Mamba_en': args.Mamba_en,
        'Fusion_en': args.Fusion_en,
        'num_heatmaps': 10,
        'heatmap_save_dir': 'figures/heatmaps',
    }
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))
    print("model-type: {} (clip/coop/clip_vit/clip_vit_dual)".format(args['model_type']))

    # 根据 --model-type 加载对应模型
    if args['model_type'] == 'clip':
        regular_model = CLIPZeroShot(args, classnames=dataset_classes, clip_model=clip_model).cuda()
    elif args['model_type'] == 'coop':
        regular_model = CoopCLIP(args, classnames=dataset_classes, clip_model=clip_model).cuda()
    elif args['model_type'] == 'clip_vit':
        regular_model = CLIPVITBase(args, classnames=dataset_classes, clip_model=clip_model).cuda()
    else:
        regular_model = CLIPVITDual(args, classnames=dataset_classes, clip_model=clip_model).cuda()
    engine = MultiLabelEngine(args)

    # clip 为零样本不加载 ckpt；其余模型加载对应 ckpt
    if args['model_type'] != 'clip' and args['resume'] and os.path.isfile(args['resume']):
        checkpoint = torch.load(args['resume'], map_location='cuda')
        state = checkpoint.get('state_dict', checkpoint)
        filtered_dict = {k.replace("module.", ""): v for k, v in state.items()}
        regular_model.load_state_dict(filtered_dict, strict=True)
    
    test_loader = torch.utils.data.DataLoader(
                                            val_dataset,
                                            batch_size=args['batch_size'],
                                            shuffle=False,
                                            num_workers = args['workers'],
                                            drop_last=False
                                        )

    regular_ap, regular_map, reg_meters, reg_topk= engine.learning(regular_model, test_loader)

    print("test mAP:{}".format(regular_map))
    head_AP, middle_AP, tail_AP, head, medium, tail = ltAnalysis(regular_ap, args['dataset'])
    filename = os.path.join('log/log_coco', str(args['dataset'])+"_"+args['resume'][args['resume'].rfind("/")+1:].replace(".ckpt","_{}.txt".format(regular_map)))
    with open(filename,'a') as f:
        f.write(str(args)+"\n")
        f.write("=================================================>>>>>>> OP, OR, OF1, CP, CR, CF1:"+"\n")
        f.write(str(reg_meters)+"\n")
        f.write("=================================================>>>>>>> OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k:"+"\n")
        f.write(str(reg_topk)+"\n")
        f.write("test mAP:"+str(regular_map)+"\n")
        f.write("head APs:"+str(head_AP)+"\n")
        f.write("middle APs:"+str(middle_AP)+"\n")
        f.write("tail APs:"+str(tail_AP)+"\n")
        f.write("=================================================>>>>>>> mAP head, mAP medium, mAP tail:"+"\n")
        f.write(str(head)+","+str(medium)+","+str(tail)+"\n")



if __name__ == '__main__':
    main_coco()
