from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
import clip

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).to(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).to(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class TextFeatureEnhancer(nn.Module):
    """文本特征增强：残差 MLP + LayerNorm"""
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
class CLIPVIT(nn.Module):

    def get_tokenized_prompts(self, classnames):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device = torch.device('cuda' if torch.cuda.is_available() else "cpu"))
        return prompts    

    def __init__(self, args, classnames, clip_model, embed_dim=768):
        super().__init__()

        self.final_dim = 512
        self.global_only = False

        self.use_dynamic_prompt = args.get('use_dynamic_prompt', 0)
        self.use_text_enhance = args.get('use_text_enhance', 0)
        self.use_dual_consistency = args.get('use_dual_consistency', 0)
        self.n_cls = len(classnames)

        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.clipzero = False

        self.use_clip_proj = False

        self.prompts = self.get_tokenized_prompts(classnames)
        self.text_encoder = TextEncoder(clip_model)

        if not self.use_clip_proj:
            self.projection = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(embed_dim, self.final_dim)),
                    ('act', nn.Tanh()),
                    ('fc2', nn.Linear(self.final_dim, self.final_dim))],)
            )

        self.projection_dist = clip_model.visual.proj
        self.topk = args['topk']

        # 创新1：动态文本提示/调制（根据图像全局特征调制文本特征）
        if self.use_dynamic_prompt:
            self.dynamic_text_net = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.n_cls * 512),
            )
            nn.init.normal_(self.dynamic_text_net[-1].weight, std=0.02)
            nn.init.zeros_(self.dynamic_text_net[-1].bias)
        # 创新2：文本特征增强
        if self.use_text_enhance:
            self.text_enhancer = TextFeatureEnhancer(dim=512, scale=0.1)

    def forward_features(self, x):
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x1 = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x2 = x


        x = torch.cat([x1, x2], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        return x

    def _get_patch_features_before_transformer(self, x):
        """得到进入 Transformer 之前的 patch 特征 x2 [B, N, C]，用于更局部的热力图。"""
        x = self.conv1(x)
        if self.Mamba_en:
            x_s = self.ss2d(x)
            x_o = x
            x = self.sim(x_s, x_o)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x1 = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x2 = self.Fusion(x1, x) if self.Fusion_en else x
        return x2

    def get_spatial_attention(self, x, class_idx=None, source='pre_transformer'):
        """
        获取 patch 级空间注意力，用于热力图可视化。

        特征位置说明：
        - source='pre_transformer'（默认）：使用 **进入 Transformer 之前** 的 patch 特征（conv + 可选 NNSBlock/Fusion 之后），
          投影到 512 维后与文本做余弦相似度。空间局部性保留较好，热力图更集中在物体区域。
        - source='final'：使用 **Transformer 最后一层 ln_post 之后** 的 patch 特征（与预测 score 同源），
          同样投影+余弦相似度。语义强但已全局混合，热力图可能更平滑。

        Args:
            x: [B, 3, H, W] 输入图像
            class_idx: 若为 int，则取该类与各 patch 的相似度；若为 None，取所有类中每 patch 的最大响应
            source: 'final' | 'pre_transformer'，见上
        Returns:
            attention: [B, h, w]，h=w=14 (ViT-B/16)
        """
        with torch.no_grad():
            if source == 'pre_transformer':
                patch_feat = self._get_patch_features_before_transformer(x)
                pred_feat = patch_feat @ self.projection_dist
            else:
                feat = self.forward_features(x)
                pred_feat = feat[:, 1:]
                pred_feat = pred_feat @ self.projection_dist
            pred_feat = F.normalize(pred_feat, p=2, dim=-1)
            text_features = self.text_encoder(self.prompts)
            text_features = text_features.float()
            text_features = F.normalize(text_features, p=2, dim=-1)
            spatial = pred_feat @ text_features.t()
            if class_idx is not None:
                att = spatial[:, :, class_idx]
            else:
                att = spatial.max(dim=2)[0]
            n = att.shape[1]
            side = int(round(n ** 0.5))
            att = att.view(att.shape[0], side, side)
        return att

    def forward(self, x, label_emb=None, norm_pred=True):
        """
        Args:
            x: input images
            label_emb: optional [N_cls, dim] label embeddings (e.g. for NUS-wide).
                       若提供则用 label_emb 计算 logits，否则用文本 encoder。
            norm_pred: 是否对 score 做归一化
        Returns:
            score (logits), pred_feat, dist_feat
        """
        x = self.forward_features(x)
        dist_feat = x[:, 0] @ self.projection_dist
        
        
            
        if label_emb is not None:

            # 使用外部 label embedding（如 NUS 训练/测试）

            if not self.use_clip_proj:
                pred_feat = x[:, 1:] @ self.projection_dist
            else:
                pred_feat = x[:, 1:] @ self.projection_dist
            score1 = torch.topk(pred_feat @ label_emb.t(), k=self.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ label_emb.t()
            if norm_pred:
                score1 = score1 / score1.norm(dim=-1, keepdim=True)
                score2 = score2 / score2.norm(dim=-1, keepdim=True)
            score = (score1 + score2) / 2
            return score, pred_feat, dist_feat

        prompts = self.prompts
        text_features = self.text_encoder(prompts)
        text_features = text_features.float()

        # 创新1：动态文本调制（用当前 batch 的均值视觉特征调制文本特征）
        if self.use_dynamic_prompt and hasattr(self, 'dynamic_text_net'):

            delta = self.dynamic_text_net(dist_feat.mean(0, keepdim=True)).view(self.n_cls, 512)
            text_features = text_features + delta.to(text_features.dtype)
        # 创新2：文本特征增强
        if self.use_text_enhance and hasattr(self, 'text_enhancer'):

            text_features = self.text_enhancer(text_features)
        # For Global Head Only Ablation
        if self.global_only:
            score = dist_feat @ text_features.t()
            if norm_pred:
                score = score / score.norm(dim=-1, keepdim=True)
            return score, x[:, 1:], dist_feat

        # Default
        else:

            if not self.use_clip_proj:
                pred_feat = x[:, 1:] @ self.projection_dist
            else:
                pred_feat = x[:, 1:] @ self.projection_dist                
            score1 = torch.topk(pred_feat @ text_features.t(), k=self.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ text_features.t()
            if norm_pred:
                score1 = score1 / score1.norm(dim=-1, keepdim=True)
                score2 = score2 / score2.norm(dim=-1, keepdim=True)
            score = (score1 + score2) / 2
            # 创新3：双分支一致性损失需要返回 score1, score2
            if self.use_dual_consistency:

                return score, pred_feat, dist_feat, score1, score2
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