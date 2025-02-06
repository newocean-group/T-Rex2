import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from ops.modules import MSDeformAttn
import numpy as np
from torch.nn.init import normal_, xavier_uniform_


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        # self attention
        self.self_attn = MSDeformAttn(
            d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.act(self.linear1(src))))
        src = src + self.dropout3(src2)
        return self.norm2(src)

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
    ):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_visual_cross_attn=True,
    ):
        super().__init__()
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # visual cross attention
        self.use_visual_cross_attention = use_visual_cross_attn
        if use_visual_cross_attn:
            self.ca_visual = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.ca_visual_dropout = nn.Dropout(dropout)
            self.ca_visual_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
        memory_visual=None,
        self_attention_mask=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
            attn_mask=self_attention_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, return_intermediate=False, d_model=256
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.d_model = d_model
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.query_scale = None

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.ref_point_head.layers[0].weight)
        xavier_uniform_(self.ref_point_head.layers[1].weight)

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        memory_visual=None,
        self_attention_mask=None,
        bbox_head=None,
        score_head=None,
    ):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []

        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points.transpose(0, 1).contiguous()[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points.transpose(0, 1).contiguous()[:, :, None]
                    * src_valid_ratios[None, :]
                )
            query_sine_embed = (
                gen_sineembed_for_position(reference_points_input[:, :, 0, :])
                .transpose(0, 1)
                .contiguous()
            )  # nq, bs, 256*2

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            # main process
            output = layer(
                output,
                query_pos,
                reference_points_input.transpose(0, 1).contiguous(),
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
                memory_visual,
                self_attention_mask,
            )

            if score_head is not None:
                if memory_visual is not None:
                    dec_out_logits.append(score_head[layer_id](output, memory_visual))
                else:
                    dec_out_logits.append(score_head[layer_id](output))

            if bbox_head is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = bbox_head[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()
                reference_points = new_reference_points.detach()
                dec_out_bboxes.append(new_reference_points)

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class DeformableTransformerLayerForVisualEncoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        # cross attn
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attn
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_padding_mask=None,
        self_pos=None,
        reference_points_mask=None,
        self_attn_mask=None,
    ):
        # cross attn
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            src_level_start_index,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # self attn
        q = k = self.with_pos_embed(tgt, self_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
            key_padding_mask=reference_points_mask,
            attn_mask=self_attn_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


class DeformableTransformerForVisualEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, nhead=8, d_model=256):
        super().__init__()
        self.n_heads = nhead
        self.layers = _get_clones(encoder_layer, num_layers)
        self.query_scale = None
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        src_padding_mask=None,
        reference_points_mask=None,
    ):
        _, L, _ = tgt.shape
        self_attn_mask = (
            reference_points_mask[:, None]
            .repeat(1, L, 1)
            .repeat_interleave(self.n_heads, dim=0)
        )
        for layer in self.layers:
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points.transpose(0, 1).contiguous()[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[None, :]
                )  # bs, nq, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points.transpose(0, 1).contiguous()[:, :, None]
                    * src_valid_ratios[None, :]
                )
            query_sine_embed = (
                gen_sineembed_for_position(reference_points_input[:, :, 0, :])
                .transpose(0, 1)
                .contiguous()
            )  # bs, nq, 256*2

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # bs, nq, 256
            pos_scale = self.query_scale(tgt) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            tgt = layer(
                tgt=tgt,
                query_pos=query_pos,
                reference_points=reference_points_input.transpose(0, 1).contiguous(),
                src=src,
                src_spatial_shapes=src_spatial_shapes,
                src_level_start_index=src_level_start_index,
                src_padding_mask=src_padding_mask,
                self_pos=None,
                reference_points_mask=reference_points_mask,
                self_attn_mask=self_attn_mask,
            )
        return tgt  # bs x K + 1 x d_model


class AggregatingFeatures(nn.Module):
    def __init__(
        self, nhead=8, d_model=256, d_ffn=1024, dropout=0.0, activation="relu"
    ):
        super().__init__()
        self.n_heads = nhead
        # self attn
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward(self, tgt):
        # self attn
        _, L, _ = tgt.shape
        q = k = tgt
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
        )[0].transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


class AggregatingFeaturesForVisualEncoder(nn.Module):
    def __init__(self, aggregating_features_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(aggregating_features_layer, num_layers)

    def forward(self, tgt):
        for layer in self.layers:
            tgt = layer(tgt)
        return tgt


class ContrastiveEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.bias = nn.Parameter(torch.zeros([]))

    def create_proj_embed_layer(self):
        self.proj = MLP(input_dim=256, hidden_dim=1024, output_dim=256, num_layers=2)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, memory, visual_embed):
        projected_visual_embed = self.proj(visual_embed)
        return (
            F.normalize(memory, p=2, dim=-1)
            @ F.normalize(projected_visual_embed, p=2, dim=-1).transpose(-1, -2)
            * self.logit_scale.exp()
            + self.bias
        )
