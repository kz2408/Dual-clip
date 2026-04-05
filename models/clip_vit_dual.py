from collections import OrderedDict

import torch
import torch.nn as nn

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts, if_embedding=False, if_sequence=False):
        if not if_embedding:
            tokenized_prompts = prompts
            prompts = self.token_embedding(prompts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # print(prompts.shape)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        
        if if_sequence:
            x = x @ self.text_projection  # NLD * Dd = NLd
            return x
        else:
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # ND * Dd = Nd
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
            return x
        
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx, ctx_init, csc, class_token_position):
        super().__init__()
        n_cls = len(classnames)
        # n_ctx = cfg.TRAINER.Caption.N_CTX
        # ctx_init = cfg.TRAINER.Caption.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            ctx_vectors_double = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            
            if csc:
                print("Initializing class-specific double contexts")
                ctx_vectors_double = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_double = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_double, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial double context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_double = nn.Parameter(ctx_vectors_double)  # to be optimized
        
        temperature = torch.tensor(3.0, dtype=dtype)  #  exp(3.91) = 50
        self.temperature = nn.Parameter(temperature)
        spatial_T = torch.tensor(3.0, dtype=dtype)  # 20
        self.spatial_T = nn.Parameter(spatial_T)
        ranking_scale = torch.tensor(4.0, dtype=dtype)  # 20
        self.ranking_scale = nn.Parameter(ranking_scale)

        # sigmoid_shift = torch.tensor(0.25, dtype=dtype)
        # self.sigmoid_shift = nn.Parameter(sigmoid_shift)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

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
        self.class_token_position = class_token_position

    def forward(self, neg_prompt_wcls=True):
        """
        Returns current learned ctx embeddings, concated with cls word embeddings.
        """
        ctx = self.ctx
        ctx_double = self.ctx_double
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

        return prompts, prompts_neg, self.temperature, self.spatial_T, self.ranking_scale


class CLIPVIT(nn.Module):

    def get_tokenized_prompts(self, classnames):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device = torch.device('cuda' if torch.cuda.is_available() else "cpu"))
        return prompts    

    # def __init__(self, args, classnames, clip_model, n_ctx, ctx_init, csc, class_token_position,embed_dim=768):
    def __init__(self, args, classnames, clip_model, embed_dim=768):
        super().__init__()

        self.final_dim = 512
        self.global_only = False
        
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.clipzero = False

        self.use_clip_proj = False

        self.prompts = self.get_tokenized_prompts(classnames)
        # self.prompt_learner = PromptLearner(classnames, clip_model, n_ctx, ctx_init, csc, class_token_position)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in self.prompts])
        self.text_encoder = TextEncoder(clip_model)

        if not self.use_clip_proj:
            self.projection = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(embed_dim, self.final_dim)),
                    ('act', nn.Tanh()),
                    ('fc2', nn.Linear(self.final_dim, self.final_dim))],)
            )

        self.projection_dist = clip_model.visual.proj
        self.topk = args['topk']
        self.ratio = args['alpha']
    
    def forward_features(self, x):


        x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        return x

    def forward(self, x, norm_pred=True):

        x = self.forward_features(x)
        dist_feat = x[:, 0] @ self.projection_dist

        # prompts, prompts_double, _,_,_= self.prompt_learner()
        prompts = self.prompts
        # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = prompts
        # tokenized_prompts = self.tokenized_prompts
        # prompts = prompts.long()
        # prompts_double = prompts_double.long()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        # text_features_neg = self.text_encoder(prompts_double, tokenized_prompts)
        text_features_neg = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features.float()
        text_features_neg = text_features_neg.float()

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
            score1 = torch.topk(pred_feat @ text_features.t(),k=self.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ text_features_neg.t()
            if norm_pred:
                score1 = score1 / score1.norm(dim=-1, keepdim=True)
                score2 = score2 / score2.norm(dim=-1, keepdim=True)
            
            # score = (score1 + score2) / 2 
            score = self.ratio * score1 + (1-self.ratio) *score2
            return score, pred_feat, dist_feat

    def encode_img(self, x):
        # import pdb; pdb.set_trace()
        x = self.forward_features(x)
        if self.clipzero:
            x = x @ self.proj
            return x[:, 1:, :], x[:, 0, :]
        else:
            pred_feat = x[:, 1:] @ self.projection_dist
            # dist_feat = self.projection_dist(x[:, 0])
            dist_feat = x[:, 0] @ self.projection_dist
            return pred_feat, dist_feat