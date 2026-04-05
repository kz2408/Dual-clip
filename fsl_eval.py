from audioop import avg
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from utils.engine_fsl_val_TF import MultiLabelEngine
from src.loss_functions.losses import AsymmetricLoss, CosLoss,AsymmetricLossOptimized
# from exp import *
torch.multiprocessing.set_sharing_strategy('file_system')
from models.prompt_model import CLIPVIT
# use_cos = True

import clip

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--dataset', help='dataset', default='coco')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--num-classes', default=16)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--threshold', default=0.5, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--ctx_init', default='a photo of a', type=str, help='init context prompt')
parser.add_argument('--n_ctx', default=4, type=int, help='length M of context prompt when initializing')
parser.add_argument('--class_token_position', default='end', type=str, help='position of class token')

parser.add_argument('--shot', default=5, type=int, choices=[1,5], help='1 or 5 shot')
parser.add_argument('--epis', nargs = '+', type=str, default=['1','2','3','4','5','6','7','8','9','10'])
# parser.add_argument('--epis', nargs = '+', type=str, default=['8','9','10'])
parser.add_argument('--pretrain_clip', default='ViT16', type=str, choices=['RN50', 'ViT16'], help='pretrained clip backbone')
parser.add_argument('--topk',type = int,default=16)
parser.add_argument('--lamda',type = float,default=1.0)
parser.add_argument('--alpha',type = float,default=0.5)

parser.add_argument('--feature', default='', type=str)

def main():
    args = parser.parse_args()
    feature = args.feature
    # hyper-parameters
    model_name = 'fsl_First_Stage_al{}_lr{}_clip_vit_topk{}_{}'.format(args.alpha,args.lr,args.topk,args.dataset)
        
    if args.dataset == 'coco':
        args.data = '/data2/yanjiexuan/coco/data'

        inp_seman = 'data/coco/coco_glove_300_coco_sequence.pkl'
        args.num_classes = 16
        dataset_classes = [
            'bicycle', 'boat', 'stop_sign', 'bird',
            'backpack', 'frisbee', 'snowboard', 'surfboard', 
            'cup', 'fork', 'spoon', 'broccoli', 
            'chair', 'keyboard', 'microwave', 'vase'
            ]  

    elif args.dataset == 'nus':
        args.data = '/data2/yanjiexuan/nuswide'
        inp_seman = 'data/nuswide/nuswide_glove_word2vec.pkl'

    elif args.dataset == 'voc':
        args.data = '/data2/yanjiexuan/voc'
        inp_seman = 'data/voc/voc_glove_word2vec.pkl'
        args.num_classes = 6
        dataset_classes = [
            'cat', 'dog', 'pottedplant', 'sheep', 'sofa', 'tvmonitor'
            ]
    else:
        raise NotImplementedError
    
    if args.pretrain_clip == "RN50":
        pretrain_clip_path = '/data2/yanjiexuan/huggingface/openai/pretrained/RN50.pt'
    elif args.pretrain_clip == "ViT16":
        pretrain_clip_path = '/data2/yanjiexuan/huggingface/openai/pretrained/ViT-B-16.pt'

    print(f"Loading CLIP (backbone: {args.pretrain_clip})")
    clip_model, preprocess = clip.load(pretrain_clip_path, device='cpu', jit=False) # Must set jit=False for training

    args = {
        'num_classes': args.num_classes,
        'max_epoch': 150,
        'resume': args.resume,
        'evaluation': args.evaluate,
        'threshold': args.threshold,
        'lr': args.lr,
        'train': False,

        'dataset':args.dataset,
        'inp_seman':inp_seman,
        'data': args.data,
        'image_size':args.image_size,
        'workers':args.workers,
        'batch_size': args.batch_size,
        'epis': args.epis,
        'shot': args.shot,
        'fix_sample':False,

        'dataset': args.dataset,
        'ctx_init': args.ctx_init,
        'n_ctx': args.n_ctx,
        'class_token_position': args.class_token_position,
        'topk': args.topk,
        'lamda': args.lamda,
        'alpha': args.alpha
    }

    print(args)
    # Setup model
    print('creating model...')

    regular_model = CLIPVIT(args,classnames=dataset_classes, clip_model=clip_model).cuda()
    # regular_model = torch.nn.DataParallel(regular_model)
    # if args['resume']:
    #     if os.path.isfile(args['resume']):
    #         print("=> loading checkpoint '{}'".format(args['resume']))
    #         checkpoint = torch.load(args['resume'])
    #         filtered_dict = {k.replace("module.",""): v for k, v in checkpoint['state_dict'].items()}
    #         regular_model.load_state_dict(filtered_dict)
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args['resume']))

    # model = torch.nn.DataParallel(model)

    
    # set optimizer
    weight_decay = 1e-4
    # loss function
    criterion = nn.ModuleList([])
    # overall classification loss function
    criterion.append(AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True))
    # criterion.append(nn.MultiLabelSoftMarginLoss(reduction='sum'))
    # criterion.append(CosLoss(args['num_classes'], 0.5))
    # criterion.append(AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True))
    # for para in regular_model.fpn.parameters():
    #     para.requires_grad = False

    for name, param in regular_model.text_encoder.named_parameters():
        param.requires_grad = False
    
    for name, param in regular_model.prompt_learner.named_parameters():
        param.requires_grad = False
    # for para in regular_model.label_emb.parameters():
    #     para.requires_grad = False
    # for para in regular_model.fc.parameters():
    #     para.requires_grad = False

    # for para in model.c_transformer.attn2.parameters():
    #     para.requires_grad = False
    # for para in model.c_transformer.layers[1].parameters():
    #     para.requires_grad = False
    # parameters = add_weight_decay(models[0], weight_decay)
    # 

    # param_dicts = diff_lr(model)
    # optimizer = torch.optim.AdamW(params=parameters, lr=args['lr'], weight_decay=0) # true wd, filter_bias_and_bn
    param_dicts = [{"params": [p for n, p in regular_model.named_parameters() if p.requires_grad]}]
    optimizer = getattr(torch.optim,'AdamW')(
            param_dicts,
            lr=args['lr'],
            betas=(0.9, 0.9), eps=1e-08, weight_decay=weight_decay)

    # Actuall Training
    engine = MultiLabelEngine(args)
    assert os.path.isfile(args['resume'])
    checkpoint = torch.load(args['resume'])
    filtered_dict = {k.replace("module.",""): v for k, v in checkpoint['state_dict'].items()}

    regular_maps, reg_meter_ls = engine.learning(regular_model, clip_model, criterion, optimizer, filtered_dict, model_name, str(args['shot']),args['dataset'],args['inp_seman'])
    if len(regular_maps) == 0:
        return 

    print(str(args['shot'])+"shot model: "+ str(args))
    print("=================================================>>>>>>> regular model mAPs for epi {}:".format(args['epis']))
    print(regular_maps)
    print("=================================================>>>>>>> OP, OR, OF1, CP, CR, CF1:")
    mean_meter = [[] for i in range(6)]
    for each in reg_meter_ls:
        print(each[0])
        for i in range(6):
            mean_meter[i].append(each[0][i])
    
    for i in range(6):
        mean_meter[i] = sum(mean_meter[i])/len(mean_meter[i])

    print("=================================================>>>>>>> OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k:")
    for each in reg_meter_ls:
        print(each[1])
    mean_ = str(sum(regular_maps).item()/len(regular_maps))

    if len(regular_maps)!=10:
        filename = os.path.join('log', "uncomplete_a"+str(args['alpha'])+"_"+str(args['shot'])+"shot_"+args['resume'][args['resume'].rfind("/")+1:].replace(".ckpt","_{}_{}.txt".format(feature, mean_)))
    else:
        filename = os.path.join('log/log_fsl', "a"+str(args['alpha'])+"_"+str(args['shot'])+"shot_"+args['resume'][args['resume'].rfind("/")+1:].replace(".ckpt","_{}_{}.txt".format(feature, mean_)))
    with open(filename,'a') as f:
        f.write(str(args['shot'])+"shot model: "+ str(args))
        f.write("\n=================================================>>>>>>> regular model mAPs for epi {}:".format(args['epis'])+"\n")
        f.write(str(regular_maps)+"\n")
        f.write("=================================================>>>>>>> OP, OR, OF1, CP, CR, CF1:"+"\n")
        for each in reg_meter_ls:
            f.write(str(each[0])+"\n")
        f.write("=================================================>>>>>>> OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k:"+"\n")
        for each in reg_meter_ls:
            f.write(str(each[1])+"\n")
        try:
            f.write("mean:"+mean_+"\n")
            f.write("mean OP, OR, OF1, CP, CR, CF1:" +"\n")
            f.write(str(mean_meter))
        except:
            pass
    print("mean:"+mean_)
    print("mean OP, OR, OF1, CP, CR, CF1:")
    print(mean_meter)

if __name__ == '__main__':
    main()
