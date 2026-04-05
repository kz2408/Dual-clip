import os
import argparse
import datetime


import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import clip
from utils.misc import *
from utils.dataset import build_dataloader
from utils.optimizer import build_optimizer
from models.clip_vit import CLIPVIT as CLIPVITBase
from models.prompt_model import CLIPVIT as CLIPVITDual
from engine_nus_first_stage import train, test

def main(args):

    setup_seed(args.seed)
    args.device = "cuda:1" if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = True

    # Init Recoder
    record_name = datetime.datetime.now().strftime('%m-%d-%H:%M:%S') + "_" + "MKT"
    args.record_path = os.path.join('logger', "first_stage", record_name)
    os.makedirs(args.record_path, exist_ok=True)
    logger = init_log(args, args.record_path)
    write_description_to_folder(os.path.join(args.record_path, "configs.txt"), args)

    # Init DataLoader
    train_dataset, val_dataset = build_dataloader(args)
    len_val_dataset = len(val_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                    args.batch_size, 
                                    shuffle=True, 
                                    num_workers=args.workers, 
                                    drop_last=True)
    val_dataloader = DataLoader(val_dataset, 
                                args.test_batch_size,
                                shuffle=False, 
                                num_workers=args.workers, 
                                drop_last=False)

    # Load Label Embedding
    label_emd_path = os.path.join(args.data_path, 'label_emb.pt')
    label_emb = torch.load(label_emd_path, map_location=args.device).to(torch.float32)
    #print(label_emb.shape)
    #print(label_emb)

    # 未提供有效路径时自动下载到 ~/.cache/clip
    clip_path = args.clip_path if (args.clip_path and os.path.isfile(args.clip_path)) else "ViT-B/16"
    # Build Model（对比实验: clip_vit / clip_vit_dual）
    clip_model, _ = clip.load(clip_path, jit=False)
    model_args = {
        'topk': args.topk,
        'alpha': getattr(args, 'alpha', 0.5),
        'Mamba_en': getattr(args, 'Mamba_en', False),
        'Fusion_en': getattr(args, 'Fusion_en', False),
        'n_ctx': getattr(args, 'n_ctx', 4),
        'ctx_init': getattr(args, 'ctx_init', ''),
        'class_token_position': getattr(args, 'class_token_position', 'end'),
    }
    # NUS-wide 使用 label_emb，类别数由 label_emb 维度决定
    n_cls = label_emb.shape[0]
    dataset_classes = ["c%d" % i for i in range(n_cls)]
    if args.model_type == 'clip_vit_dual':
        model = CLIPVITDual(model_args, dataset_classes, clip_model)
    else:
        model = CLIPVITBase(model_args, dataset_classes, clip_model)
    convert_models_to_fp32(model)
    model = model.to(args.device)

    # Load CLIP
    clip_model, _ = clip.load(clip_path, jit=False)
    clip_model.eval()
    clip_model = clip_model.to(args.device)

    # Build Optimizer
    optimizer = build_optimizer(args, model)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        #pretrained_dict = torch.load('/home/liubeiyan/first_stage_best_model_nus.pth')
        #model_dict = model.state_dict()
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #model_dict.update(pretrained_dict)
        #model.load_state_dict(model_dict)
        train(model, clip_model, args, optimizer, train_dataloader, logger, label_emb, epoch)
        model.eval()
        test(model, args, val_dataloader, logger, label_emb, len_val_dataset, epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",                   type=int,   default=59  )
    parser.add_argument("--record_path",            type=str,   default='/home/liubeiyan/logger/')

    parser.add_argument("--clip-path",              type=str,   default=None, help='CLIP 权重路径；未指定或文件不存在时自动下载 ViT-B/16')
    parser.add_argument("--data-path",              type=str,   default='/data/liubeiyan/nus-wide/')
    
    parser.add_argument("--batch-size",             type=int,   default=128,     )
    parser.add_argument("--test-batch-size",        type=int,   default=471,    )
    parser.add_argument("--epochs",                 type=int,   default=20,     )
    parser.add_argument("--warmup_epochs",          type=int,   default=2,      )
    parser.add_argument("--lr",                     type=float, default=1e-3,   )
    parser.add_argument("--min_lr",                 type=float, default=1e-6,   )
    parser.add_argument("--weight_decay",           type=float, default=0.05,   )
    parser.add_argument("--workers",                type=int,   default=1,      )
    parser.add_argument("--momentum",               type=float, default=0.95,   )

    parser.add_argument("--input_size",             type=int,   default=224     )
    
    parser.add_argument("--layer_decay",            type=float, default=0.65    )
    parser.add_argument("--fix_layer",              type=int,   default=10      )
    parser.add_argument("--topk",                   type=int,   default=18      )
    parser.add_argument("--model-type",             type=str,   default='clip_vit',
                        choices=['clip_vit', 'clip_vit_dual'],
                        help='对比实验: clip_vit=ViT+模板, clip_vit_dual=双视图+可学习prompt')
    parser.add_argument("--alpha",                 type=float, default=0.5      )
    parser.add_argument("--Mamba_en",              type=bool,  default=False   )
    parser.add_argument("--Fusion_en",             type=bool,  default=False   )
    parser.add_argument("--n_ctx",                 type=int,   default=4        )
    parser.add_argument("--ctx_init",              type=str,   default=''      )
    parser.add_argument("--class_token_position",  type=str,   default='end'    )

    args = parser.parse_args()
    torch.cuda.set_device(1)
    main(args)
    

