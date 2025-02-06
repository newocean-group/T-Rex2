import math
import torch
import torch.nn as nn
from ops.modules import MSDeformAttn
from src.common_modules import (
    DeformableTransformerDecoderLayer,
    DeformableTransformerDecoder,
    ContrastiveEmbed,
    MLP,
)
from torch.nn.init import normal_, constant_, xavier_uniform_
from src.dn import *


class BoxDecoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        nhead=8,
        num_decoder_layers=6,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=4,
        dec_n_points=4,
        num_proposals=900,
        num_classes=365,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_proposals = num_proposals
        self.return_intermediate_dec = return_intermediate_dec

        # encoder head
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.enc_score_contrastive_head = ContrastiveEmbed()  # for visual prompts
        # self.enc_score_head = nn.Linear(d_model, num_classes)  # for pretrain model
        self.class_embeddings = nn.Embedding(
            num_embeddings=num_classes, embedding_dim=d_model
        )
        xavier_uniform_(self.class_embeddings.weight.data)
        self.enc_bbox_head = MLP(d_model, d_model, 4, 3)

        # decoder head
        self.dec_score_contrastive_head = nn.ModuleList(
            [ContrastiveEmbed() for _ in range(num_decoder_layers)]
        )
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(d_model, num_classes) for _ in range(num_decoder_layers)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [MLP(d_model, d_model, 4, num_layers=3) for _ in range(num_decoder_layers)]
        )

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=dec_n_points,
            use_visual_cross_attn=True,
        )

        self.decoder = DeformableTransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            return_intermediate=return_intermediate_dec,
        )

        self.tgt_embed = nn.Embedding(
            num_embeddings=num_proposals, embedding_dim=d_model
        )
        xavier_uniform_(self.tgt_embed.weight.data)

        # for denoising training
        self.denoising_class_embed = nn.Embedding(
            num_classes + 1, d_model, padding_idx=num_classes
        )
        self.num_classes = num_classes
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        constant_(self.enc_bbox_head.layers[-1].weight, 0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for reg_ in self.dec_bbox_head:
            constant_(reg_.layers[-1].weight, 0)
            constant_(reg_.layers[-1].bias, 0)

        xavier_uniform_(self.enc_output.weight)

    def create_proj_layers(self):
        self.enc_score_contrastive_head.create_proj_embed_layer()
        for m in self.dec_score_contrastive_head:
            m.create_proj_embed_layer()

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device
        )
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_mask):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_mask):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(
                N_, H_, W_, 1
            )
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], dim=1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], dim=1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H_ - 1, H_, dtype=torch.float32, device=memory.device
                ),
                torch.linspace(
                    0, W_ - 1, W_, dtype=torch.float32, device=memory.device
                ),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], dim=-1)
            scale = torch.cat(
                [valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], dim=1
            ).view(N_, 1, 1, 2)
            grid = (grid[None].expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat([grid, wh], dim=-1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, dim=1)
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )
        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        return output_memory, output_proposals

    def _get_decoder_input(
        self,
        memory,
        mask_flatten,
        spatial_shapes,
        denoising_class=None,
        denoising_bbox_unact=None,
        visual_embed=None,
    ):
        bs, _, _ = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        if visual_embed is not None:
            enc_outputs_class = self.enc_score_contrastive_head(
                output_memory, visual_embed
            )
        else:
            # enc_outputs_class = self.enc_score_head(output_memory)
            enc_outputs_class = self.enc_score_contrastive_head(
                output_memory, self.class_embeddings.weight[None].repeat(bs, 1, 1)
            )
            # old code
            # enc_outputs_class = enc_outputs_class.max(dim=-1, keepdim=True)[0]

        # new code
        enc_outputs_class = enc_outputs_class.max(dim=-1, keepdim=True)[0]

        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + output_proposals

        topk_proposals = torch.topk(
            enc_outputs_class[..., 0], self.num_proposals, dim=1
        )[1]

        refpoint_embed_undetach = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoid

        if denoising_bbox_unact is not None:
            refpoint_embed_undetach = torch.concat(
                [denoising_bbox_unact, refpoint_embed_undetach], dim=1
            )

        refpoint_embed_ = refpoint_embed_undetach.detach()

        refpoint_embed = refpoint_embed_.sigmoid()  # bs x num_proposals x 4

        tgt = self.tgt_embed.weight[None].repeat(
            bs, 1, 1
        )  # bs x num_proposals x d_model

        if denoising_class is not None:
            tgt = torch.concat([denoising_class, tgt], 1)

        return tgt, refpoint_embed

    def forward_for_text_promtps(
        self,
        memory,
        mask_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        text_embeds,
    ):
        bs, _, _ = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        enc_outputs_class = self.enc_score_contrastive_head(
            output_memory, text_embeds[None]
        )
        enc_outputs_class = enc_outputs_class.max(dim=-1, keepdim=True)[0]

        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + output_proposals

        topk_proposals = torch.topk(
            enc_outputs_class[..., 0], self.num_proposals, dim=1
        )[1]

        refpoint_embed_undetach = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoid

        refpoint_embed_ = refpoint_embed_undetach.detach()

        refpoint_embed = refpoint_embed_.sigmoid()  # bs x num_proposals x 4

        tgt = self.tgt_embed.weight[None].repeat(
            bs, 1, 1
        )  # bs x num_proposals x d_model

        out_bboxes, out_logits = self.decoder(
            tgt=tgt,
            reference_points=refpoint_embed,
            src=memory,
            src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index,
            src_valid_ratios=valid_ratios,
            query_pos=None,
            src_padding_mask=mask_flatten,
            memory_visual=text_embeds[None],
            self_attention_mask=None,
            bbox_head=self.dec_bbox_head,
            score_head=self.dec_score_contrastive_head,
        )

        return out_bboxes, out_logits, None, None, None

    def forward(
        self,
        memory,
        mask_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        visual_embed=None,
        targets=None,
    ):
        bs, _, _ = memory.shape
        if targets is not None:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_proposals,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                )
            )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = (
                None,
                None,
                None,
                None,
            )

        # prepare for dn
        tgt, refpoint_embed = self._get_decoder_input(
            memory,
            mask_flatten,
            spatial_shapes,
            denoising_class,
            denoising_bbox_unact,
            visual_embed,
        )

        # main process
        visual_embed = (
            self.class_embeddings.weight[None].repeat(bs, 1, 1)
            if visual_embed is None
            else visual_embed
        )

        out_bboxes, out_logits = self.decoder(
            tgt=tgt,
            reference_points=refpoint_embed,
            src=memory,
            src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index,
            src_valid_ratios=valid_ratios,
            query_pos=None,
            src_padding_mask=mask_flatten,
            memory_visual=visual_embed,
            self_attention_mask=attn_mask,
            bbox_head=self.dec_bbox_head,
            # score_head=(
            #     self.dec_score_head
            #     if visual_embed is None
            #     else self.dec_score_contrastive_head
            # ),
            score_head=self.dec_score_contrastive_head,
        )

        if dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta["dn_num_split"], dim=2
            )
            return out_bboxes, out_logits, dn_out_bboxes, dn_out_logits, dn_meta

        return out_bboxes, out_logits, None, None, None

    def _get_decoder_input_for_training(
        self,
        memory,
        mask_flatten,
        spatial_shapes,
        denoising_class=None,
        denoising_bbox_unact=None,
        class_embeddings=None,
    ):
        bs, _, _ = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        enc_outputs_class = self.enc_score_contrastive_head(
            output_memory, class_embeddings.weight[None].repeat(bs, 1, 1)
        )

        enc_outputs_class = enc_outputs_class.max(dim=-1, keepdim=True)[0]

        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + output_proposals

        topk_proposals = torch.topk(
            enc_outputs_class[..., 0], self.num_proposals, dim=1
        )[1]

        refpoint_embed_undetach = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoid

        if denoising_bbox_unact is not None:
            refpoint_embed_undetach = torch.concat(
                [denoising_bbox_unact, refpoint_embed_undetach], dim=1
            )

        refpoint_embed_ = refpoint_embed_undetach.detach()

        refpoint_embed = refpoint_embed_.sigmoid()  # bs x num_proposals x 4

        tgt = self.tgt_embed.weight[None].repeat(
            bs, 1, 1
        )  # bs x num_proposals x d_model

        if denoising_class is not None:
            tgt = torch.concat([denoising_class, tgt], 1)

        return tgt, refpoint_embed

    def forward_for_training(
        self,
        memory,
        mask_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        visual_embed=None,
        targets=None,
        num_classes=None,
        class_embeddings=None,
        denoising_class_embed=None,
    ):
        bs, _, _ = memory.shape
        denoising_class, denoising_bbox_unact, attn_mask, dn_meta = (
            get_contrastive_denoising_training_group(
                targets,
                num_classes,
                self.num_proposals,
                denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )
        )

        # prepare for dn
        tgt, refpoint_embed = self._get_decoder_input_for_training(
            memory,
            mask_flatten,
            spatial_shapes,
            denoising_class,
            denoising_bbox_unact,
            class_embeddings,
        )

        # main process
        visual_embed = (
            class_embeddings.weight[None].repeat(bs, 1, 1)
            if visual_embed is None
            else visual_embed
        )

        out_bboxes, out_logits = self.decoder(
            tgt=tgt,
            reference_points=refpoint_embed,
            src=memory,
            src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index,
            src_valid_ratios=valid_ratios,
            query_pos=None,
            src_padding_mask=mask_flatten,
            memory_visual=visual_embed,
            self_attention_mask=attn_mask,
            bbox_head=self.dec_bbox_head,
            score_head=self.dec_score_contrastive_head,
        )

        if dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta["dn_num_split"], dim=2
            )
            return out_bboxes, out_logits, dn_out_bboxes, dn_out_logits, dn_meta

        return out_bboxes, out_logits, None, None, None
