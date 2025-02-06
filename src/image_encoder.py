import torch
import torch.nn as nn
from ops.modules import MSDeformAttn
from src.common_modules import (
    DeformableTransformerEncoderLayer,
    DeformableTransformerEncoder,
)
from src.backbone import build_backbone
from src.utils import NestedTensor
import torch.nn.functional as F
from torch.nn.init import normal_


class ImageEncoder(nn.Module):
    def __init__(
        self,
        model_name="swin_L_384_22k",
        pretrain_img_size=1024,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        d_ffn=1024,
        dropout=0.0,
        activation="relu",
        num_feature_levels=4,
        enc_n_points=4,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels

        self.backbone, self.backbone_out_channels = build_backbone(
            model_name=model_name, pretrain_img_size=pretrain_img_size
        )

        num_backbone_outs = len(self.backbone_out_channels)

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=enc_n_points,
        )

        self.encoder = DeformableTransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        input_proj_list = []
        for in_channels in self.backbone_out_channels:
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                )
            )
        for _ in range(num_feature_levels - num_backbone_outs):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.backbone_out_channels[-1],
                        d_model,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(32, d_model),
                )
            )

        self.input_proj = nn.ModuleList(input_proj_list)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], dim=1)
        valid_W = torch.sum(~mask[:, 0, :], dim=1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], dim=-1)
        return valid_ratio

    def forward(self, tensor_list: NestedTensor):
        # extract feats
        outputs, pos_embeds = self.backbone(tensor_list)
        srcs = [out.tensors for out in outputs]
        masks = [out.mask for out in outputs]

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed, proj) in enumerate(
            zip(srcs, masks, pos_embeds, self.input_proj)
        ):
            src = proj(src)
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1).to(torch.bool)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        if self.num_feature_levels > len(srcs):
            src = self.input_proj[-1](srcs[-1])
            mask = F.interpolate(masks[-1][None].float(), size=src.shape[-2:]).to(
                torch.bool
            )[0]
            pos_embed = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            srcs.append(src)
            masks.append(mask)
            pos_embeds.append(pos_embed)
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embed[-1].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)  # bs, L, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, L
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # memory
        memory = self.encoder(
            src=src_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            pos=lvl_pos_embed_flatten,
            padding_mask=mask_flatten,
        )

        return memory, mask_flatten, spatial_shapes, level_start_index, valid_ratios
