
from thop import profile
import os
import datetime
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import clip

from models.prompt_model import PromptLearner
from models.clip_vit import CLIPVIT


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

##视觉编码器计算量统计
# def main(args):
#
#     args.device = "cuda" if torch.cuda.is_available() else "cpu"
#     cudnn.benchmark = True
#
#     label_emb=torch.rand(7186,512).cuda()
#     print(label_emb.shape)
#
#     # Build Model
#     clip_model, _ = clip.load(args.clip_path, jit=False)
#     model = CLIPVIT(args, clip_model)
#     convert_models_to_fp32(model)
#
#     dummy_input = torch.randn(1, 3, 224, 224).cuda()
#     flops, params = profile(model, (dummy_input,label_emb))
#     print('flops: ', flops, 'params: ', params)
#     print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))





def train(text_encoder, image_encoder):


            train_inputs=torch.rand(1,3,224,224)
            train_labels=torch.rand(7186,512)

            train_inputs = train_inputs.cuda()
            train_labels = train_labels.cuda()

            txt_feat = text_encoder("seen")

            pred_feat, dist_feat = image_encoder.encode_img(train_inputs)

            score1 = torch.topk(pred_feat @ txt_feat.t(), k=image_encoder.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ txt_feat.t()
            score1 = score1 / score1.norm(dim=-1, keepdim=True)
            score2 = score2 / score2.norm(dim=-1, keepdim=True)
            logits = (score1 + score2) / 2




def main(args):
    if not args.eval:

        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"



        cudnn.benchmark = True  # For speed i.e, cudnn autotuner

        # Init Language Backbone
        text_encoder = PromptLearner(args)
        text_encoder = text_encoder.to(args.device)

        text_encoder.init_label_emb()
        convert_models_to_fp32(text_encoder)


        train_param = []
        for name, param in text_encoder.named_parameters():
            if "token_embedding" in name:
                train_param.append(param)
            else:
                param.requires_grad = False
        txt_feat = text_encoder("seen")

        flops, params = profile(text_encoder, inputs=('seen',))
        print('flops: ', flops, 'params: ', params)
        print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval", action="store_true")

    parser.add_argument("--ckpt-path", type=str, default='/home/liubeiyan/MKT/logger/first_stage/06-09-05:08:53_MKT/model_epoch_1.pth')
    parser.add_argument("--clip-path", type=str, default='/home/liubeiyan/ViT-B-16.pt')
    parser.add_argument("--eval-ckpt", type=str, default=None)
    parser.add_argument("--data-path", type=str, default='C:/Users/asus/Desktop/')

    parser.add_argument("--batch-size", type=int, default=1, )
    parser.add_argument("--test-batch-size", type=int, default=171, )
    parser.add_argument("--epochs", type=int, default=4, )
    parser.add_argument("--warmup_epochs", type=int, default=0, )
    parser.add_argument("--lr", type=float, default=1e-3, )
    parser.add_argument("--min_lr", type=float, default=1e-8, )
    parser.add_argument("--weight_decay", type=float, default=0.05, )
    parser.add_argument("--workers", type=int, default=8, )
    parser.add_argument("--momentum", type=float, default=0.95, )

    parser.add_argument("--input_size", type=int, default=224)

    parser.add_argument("--bert-embed-dim", type=int, default=512, )
    parser.add_argument("--context-length", type=int, default=77, )
    parser.add_argument("--vocab-size", type=int, default=49408, )
    parser.add_argument("--transformer-width", type=int, default=512, )
    parser.add_argument("--transformer-heads", type=int, default=8, )
    parser.add_argument("--transformer-layers", type=int, default=12, )
    parser.add_argument("--topk", type=int, default=16)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--dist_url", type=str, default='env://')

    args = parser.parse_args()
    #torch.cuda.set_device(0)
    os.environ['RANK'] = str(0)
    os.environ['WORLD_SIZE'] = str(-1)
    os.environ['LOCAL_RANK'] = str(1)
    main(args)

