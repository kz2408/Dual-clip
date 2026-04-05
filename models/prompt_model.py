import torch
import torch.nn as nn
from torch.nn import functional as F
import clip

from collections import OrderedDict


device = "cuda" if torch.cuda.is_available() else "cpu"

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts,if_embedding):
        # prompts.shape = [n_cls, ctx_length, ctx_dim]
        # tokenized_prompts.shape = [n_cls, ctx_length]
        if not if_embedding:
            tokenized_prompts = prompts
            prompts = self.token_embedding(prompts).type(self.dtype)    # [batch_size, n_ctx, d_model], [80,77,512]
        x = prompts + self.positional_embedding.type(self.dtype)        # 位置编码直接赋可学习变量，添加位置信息[80,77,512]
        x = x.permute(1, 0, 2)  # NLD -> LND  [77,80,512]
        x = self.transformer(x) #共11层，和图像encode结构一致 [77,80,512]
        x = x.permute(1, 0, 2)  # LND -> NLD  [80,77,512]
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection # [n_cls, ctx_dim] [80,512]
        # patch_text = x[:, 1:] @ self.text_projection
        # pre_text = x[:, 0] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = args['ctx_init']

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        if args['ctx_init']:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            # ctx_instance = 'a photo of a photo of'
            # n_ctx_instance = len(ctx_instance.split(" "))
            prompt = clip.tokenize(ctx_init) # [1, context_length]
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype) # [1, context_length, ctx_dim]
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :] # [n_ctx, ctx_dim]
            ctx_vectors_double = embedding[0, 1 : 1 + n_ctx, :] # [n_ctx, ctx_dim]
            prompt_prefix = ctx_init

        else:
            # # random initialization
            # if args.csc is True:
            #     print("Initializing class-specific contexts")
            #     n_ctx = args['n_ctx']
            #     ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            # else:
            #     print("Initializing a generic context")
            #     n_ctx = args['n_ctx']
            #     ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            # nn.init.normal_(ctx_vectors, std=0.02)

            # if args.csc is True:
            #     print("Initializing class-specific contexts")
            #     n_ctx = args['n_ctx']
            #     ctx_vectors_double = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            # else:
            #     print("Initializing a generic context")
            #     n_ctx = args['n_ctx']
            #     ctx_vectors_double = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            print("Initializing a generic context")
            n_ctx = args['n_ctx']
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            ctx_vectors_double = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_double, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial double context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized [n_ctx, ctx_dim]
        self.ctx_double = nn.Parameter(ctx_vectors_double)  # to be optimized [n_ctx, ctx_dim]

        temperature = torch.tensor(3.0, dtype=dtype)  #  exp(3.91) = 50
        self.temperature = nn.Parameter(temperature)
        spatial_T = torch.tensor(3.0, dtype=dtype)  # 20
        self.spatial_T = nn.Parameter(spatial_T)
        ranking_scale = torch.tensor(4.0, dtype=dtype)  # 20
        self.ranking_scale = nn.Parameter(ranking_scale)

        # # sigmoid_shift = torch.tensor(0.25, dtype=dtype)
        # # self.sigmoid_shift = nn.Parameter(sigmoid_shift)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(clip.tokenize(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # [n_cls, context_length]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # [n_cls, context_length, ctx_dim]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        # class agnostic token suffix
        prompts_nocls = [prompt_prefix + "."] * len(classnames)
        tokenized_prompts_nocls = torch.cat([clip.tokenize(p) for p in prompts_nocls])
        with torch.no_grad():
            embedding_nocls = clip_model.token_embedding(tokenized_prompts_nocls).type(dtype)
        self.register_buffer("token_suffix_nocls", embedding_nocls[:, 1 + n_ctx :, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args['class_token_position']

        # 创新1：动态文本提示词 - 根据图像全局特征调制 context
        self.use_dynamic_prompt = args.get('use_dynamic_prompt', 0)
        if self.use_dynamic_prompt:
            # 输入：视觉全局特征 512维 -> 输出：context 残差 (n_ctx * ctx_dim)
            self.dynamic_ctx_net = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, n_ctx * ctx_dim),
            )
            nn.init.normal_(self.dynamic_ctx_net[-1].weight, std=0.02)
            nn.init.zeros_(self.dynamic_ctx_net[-1].bias)

    def forward(self, neg_prompt_wcls=True, visual_feat=None):
        ctx = self.ctx
        ctx_double = self.ctx_double

        # 动态提示：用当前 batch 的均值视觉特征调制 context
        if visual_feat is not None and getattr(self, 'use_dynamic_prompt', 0) and hasattr(self, 'dynamic_ctx_net'):
            # visual_feat: (B, 512) -> delta: (1, n_ctx, ctx_dim)
            delta = self.dynamic_ctx_net(visual_feat.mean(0, keepdim=True)).view(1, self.n_ctx, -1)
            ctx = ctx + delta.to(ctx.dtype)
            ctx_double = ctx_double + delta.to(ctx_double.dtype)

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if ctx_double.dim() == 2:
            ctx_double = ctx_double.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        suffix_nocls = self.token_suffix_nocls

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

            if neg_prompt_wcls:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,     # (n_cls, n_ctx, dim)
                        suffix_nocls,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        # return prompts, prompts_neg # (n_cls, n_ctx, dim)
        return prompts, prompts_neg, self.temperature, self.spatial_T
class TextFeatureEnhancer(nn.Module):
    """创新2：文本特征增强 - 残差 MLP + LayerNorm 增强类名文本特征"""
    def __init__(self, dim=512, scale=0.1):
        super().__init__()
        self.scale = scale
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight, gain=0.01)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[3].weight, gain=0.01)
        nn.init.zeros_(self.mlp[3].bias)

    def forward(self, x):
        return x + self.scale * self.mlp(x)


class SimNet(nn.Module):
    """
    固定通道 C，任意 H/W 输入输出同尺寸
    """
    def __init__(self, dim=2048, part_num=6):
        super().__init__()
        self.temperature = 0.07

    def forward(self, x1,x2):
        x1_norm = F.normalize(x1, p=2, dim=1)  # (B, C, H, W)
        x2_norm = F.normalize(x2, p=2, dim=1)  # (B, C, H, W)

        similarity_map = (x1_norm * x2_norm).sum(dim=1, keepdim=True)

        similarity_weights = torch.sigmoid(similarity_map / self.temperature)

        x = x2 * (1 + 0.1 * similarity_weights)  # (B, C, H, W)s

        return x
class VisualEncoder(nn.Module):
    def __init__(self,clip_model,Mamba_en = False):
        super().__init__()

        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post


    def forward(self, x):
        # x = [1,3,224,224]

        x = self.conv1(x)  # shape = [*, width, grid, grid] #将图片分成[32,32]个patch [1,768,7,7]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2],合并高宽 [1,768,49]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]  更换位置 [1,49,768]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)  #[1,50,768]

        x = x.permute(1, 0, 2)  # NLD -> LND # [pixel, b, d_model] = [50, 1, 768]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD  # [1,50,768]
        x = self.ln_post(x)  # x[:, 0, :] 将所有信息汇聚到cls token,下游任务[1,768]

        return x


class CLIPVIT(nn.Module):

    def get_tokenized_prompts(self, classnames):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device = torch.device('cuda' if torch.cuda.is_available() else "cpu"))
        # text_template_features = self.text_encoder(prompts)
        return prompts
    
    def __init__(self, args, classnames, clip_model, embed_dim=768):
        super().__init__()

        self.final_dim = 512
        self.global_only = False
        self.clipzero = False

        self.use_dynamic_prompt = args.get('use_dynamic_prompt', 0)
        self.use_text_enhance = args.get('use_text_enhance', 0)
        self.use_dual_consistency = args.get('use_dual_consistency', 0)



        self.use_clip_proj = False

        self.visual_encoder = VisualEncoder(clip_model,Mamba_en = self.Mamba_en)

        self.prompt_learner = PromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts # [n_cls, ctx_length]
        # self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.template_prompts = self.get_tokenized_prompts(classnames)

        if not self.use_clip_proj:
            self.projection = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(embed_dim, self.final_dim)),
                    ('act', nn.Tanh()),
                    ('fc2', nn.Linear(self.final_dim, self.final_dim))],)
            )

        self.projection_dist = clip_model.visual.proj  #self.proj是可学习参数，维度为[768,512]
        self.topk = args['topk']
        self.alpha = args['alpha']

        # 创新2：文本特征增强
        if self.use_text_enhance:
            self.text_enhancer = TextFeatureEnhancer(dim=512, scale=0.1)

    def forward(self, x, label_emb=None, norm_pred=True):
        """
        Args:
            x: input images
            label_emb: optional [N_cls, dim] label embeddings (e.g. for NUS-wide).
                       若提供则用 label_emb 计算 logits，返回 (score, pred_feat, dist_feat) 三元组。
            norm_pred: 是否对 score 做归一化
        """
        x = self.visual_encoder(x)
        dist_feat = x[:, 0] @ self.projection_dist

        # 创新1：动态文本提示需要先有 dist_feat，再生成 prompt
        visual_feat_for_prompt = dist_feat if self.use_dynamic_prompt else None

        if label_emb is not None:
            # 使用外部 label embedding（如 NUS 训练）
            if not self.use_clip_proj:
                pred_feat = x[:, 1:] @ self.projection_dist
            else:
                pred_feat = x[:, 1:] @ self.projection_dist
            if self.Fusion_en:
                pred_feat = self.Fusion(dist_feat, pred_feat)
            score1 = torch.topk(pred_feat @ label_emb.t(), k=self.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ label_emb.t()
            if norm_pred:
                score1 = score1 / score1.norm(dim=-1, keepdim=True)
                score2 = score2 / score2.norm(dim=-1, keepdim=True)
            score = self.alpha * score1 + (1 - self.alpha) * score2
            return score, pred_feat, dist_feat

        # prompts, prompts_double = self.prompt_learner()  # [n_cls, ctx_length, ctx_dim]
        prompts, prompts_double, temperature, spatial_T = self.prompt_learner(visual_feat=visual_feat_for_prompt)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts, if_embedding=True)
        text_features_neg = self.text_encoder(prompts_double, tokenized_prompts, if_embedding=True)
        text_features = text_features.float()
        text_features_neg = text_features_neg.float()
        # 创新2：文本特征增强
        if self.use_text_enhance:
            text_features = self.text_enhancer(text_features)
            text_features_neg = self.text_enhancer(text_features_neg)

        template_prompts = self.template_prompts
        template_text_features = self.text_encoder(template_prompts,template_prompts,if_embedding = False)
        # print(text_features.shape)
        template_text_features = template_text_features.float()

        # logit_scale = self.logit_scale.exp()
        logit_scale = temperature.exp()  # rk_scale
        tmp_scale = spatial_T.exp()

        # For Global Head Only Ablation
        if self.global_only:
            score = dist_feat @ text_features.t()
            if norm_pred:
                score = score / score.norm(dim=-1, keepdim=True)
            return score, x[:, 1:], dist_feat

        # Default
        else:
            if not self.use_clip_proj:
            # pred_feat = self.projection(x[:, 1:])
                pred_feat = x[:, 1:] @ self.projection_dist
            else:
                pred_feat = x[:, 1:] @ self.projection_dist

            if self.Fusion_en:

                pred_feat = self.Fusion(dist_feat,pred_feat) 

            

            score1 = torch.topk(logit_scale * pred_feat @ text_features.t(), k=self.topk, dim=1)[0].mean(dim=1)
            score2 = tmp_scale * dist_feat @ text_features_neg.t()
            if norm_pred:
                score1 = score1 / score1.norm(dim=-1, keepdim=True)
                score2 = score2 / score2.norm(dim=-1, keepdim=True)

            score = self.alpha * score1 + (1 - self.alpha) * score2
            # 创新3：双分支局部与全局一致性损失需要返回 score1/score2 供 engine 计算
            if self.use_dual_consistency:
                return text_features, template_text_features, score, pred_feat, dist_feat, score1, score2
            return text_features, template_text_features, score, pred_feat, dist_feat

        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)
        
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        # logits_neg = logit_scale * image_features @ text_features_neg.t()
    def encode_img(self, x):
        # import pdb; pdb.set_trace()
        x = self.visual_encoder(x)
        if self.clipzero:
            x = x @ self.proj
            return x[:, 1:, :], x[:, 0, :]
        else:
            pred_feat = x[:, 1:] @ self.projection_dist
            # dist_feat = self.projection_dist(x[:, 0])
            dist_feat = x[:, 0] @ self.projection_dist
            return pred_feat, dist_feat
        
   
