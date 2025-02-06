from src.image_encoder import ImageEncoder
from src.visual_encoder import VisualPromptEncoder
from src.box_decoder import BoxDecoder
import torch.nn as nn
from src.utils import NestedTensor
import torch
from typing import List


class TRex2(nn.Module):
    def __init__(
        self,
        model_name="swin_L_384_22k",
        pretrain_img_size=1024,
        d_model=256,
        d_ffn=1024,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_layers_for_visual_prompt=3,
        dropout=0.0,
        activation="relu",
        num_features_levels=4,
        enc_n_points=4,
        dec_n_points=4,
        visual_n_points=4,
        num_proposals=900,
        return_intermediate_dec=True,
        num_classes=365,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            model_name=model_name,
            pretrain_img_size=pretrain_img_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            num_feature_levels=num_features_levels,
            enc_n_points=enc_n_points,
        )

        self.visual_prompt_encoder = VisualPromptEncoder(
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            num_layers=num_layers_for_visual_prompt,
            n_points=visual_n_points,
            dropout=dropout,
            n_levels=num_features_levels,
        )

        self.box_decoder = BoxDecoder(
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            activation=activation,
            return_intermediate_dec=return_intermediate_dec,
            dec_n_points=dec_n_points,
            num_proposals=num_proposals,
            num_classes=num_classes,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
        )

    def create_logit_scale_and_bias(self):
        count = 0
        for m in self.box_decoder.dec_score_contrastive_head:
            m.create_logit_scale_and_bias()
            count += 1
        self.box_decoder.enc_score_contrastive_head.create_logit_scale_and_bias()
        count += 1
        print("logit_scale_and_bias count: ", count)

    def inverse_sigmoid(self, x, eps=1e-8):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    def get_memory(self, tensor_list: NestedTensor):
        # memory
        memory, mask_flatten, spatial_shapes, level_start_index, valid_ratios = (
            self.image_encoder(tensor_list)
        )
        return memory, mask_flatten, spatial_shapes, level_start_index, valid_ratios

    def get_visual_embed_with_memory(
        self,
        memory,
        mask_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        reference_tensor_list: NestedTensor,
    ):
        # visual embed
        reference_points = reference_tensor_list.tensors
        reference_points_mask = reference_tensor_list.mask
        visual_embed = self.visual_prompt_encoder(
            memory=memory,
            mask_flatten=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reference_points=reference_points,
            reference_points_mask=reference_points_mask,
        )

        return visual_embed

    def get_output_with_memory_for_text(
        self,
        memory,
        mask_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        text_embed,
    ):
        # box decoder
        out_bboxes, out_logits, dn_out_bboxes, dn_out_logits, dn_meta = (
            self.box_decoder(
                memory=memory,
                mask_flatten=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                visual_embed=text_embed,
                targets=None,
            )
        )
        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
        if dn_meta is not None:

            out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
            out["dn_meta"] = dn_meta
        return out, text_embed

    def get_output_with_memory(
        self,
        memory,
        mask_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        visual_embed,
    ):

        # box decoder
        out_bboxes, out_logits, dn_out_bboxes, dn_out_logits, dn_meta = (
            self.box_decoder(
                memory=memory,
                mask_flatten=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                visual_embed=visual_embed[:, -1:] if visual_embed is not None else None,
                targets=None,
            )
        )
        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
        if dn_meta is not None:

            out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
            out["dn_meta"] = dn_meta
        return out, visual_embed

    def forward_visual_encoder(
        self, tensor_list: NestedTensor, reference_tensor_list: NestedTensor
    ):
        # memory
        memory, mask_flatten, spatial_shapes, level_start_index, valid_ratios = (
            self.image_encoder(tensor_list)
        )

        # visual embed
        reference_points = reference_tensor_list.tensors
        reference_points_mask = reference_tensor_list.mask
        visual_embed = self.visual_prompt_encoder(
            memory=memory,
            mask_flatten=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reference_points=reference_points,
            reference_points_mask=reference_points_mask,
        )

        return visual_embed

    def get_outputs_with_multi_visual_embeds(
        self,
        memory,
        mask_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        multi_visual_embeds,
    ):
        # outputs
        # box decoder
        out_bboxes, out_logits, dn_out_bboxes, dn_out_logits, dn_meta = (
            self.box_decoder(
                memory=memory,
                mask_flatten=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                visual_embed=multi_visual_embeds,
                targets=None,
            )
        )

        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
        if dn_meta is not None:
            out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
            out["dn_meta"] = dn_meta
        return out, multi_visual_embeds

    def get_outputs_with_targets_and_memory(
        self,
        targets,
        memory,
        mask_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
    ):
        # visual embed
        visual_embed = None

        # outputs
        # box decoder
        out_bboxes, out_logits, dn_out_bboxes, dn_out_logits, dn_meta = (
            self.box_decoder(
                memory=memory,
                mask_flatten=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                visual_embed=visual_embed,
                targets=targets,
            )
        )

        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
        if dn_meta is not None:
            out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
            out["dn_meta"] = dn_meta
        return out, visual_embed

    def forward_for_training(
        self,
        tensor_list: NestedTensor,
        reference_tensor_list_list: List[NestedTensor],
        targets,
        num_classes,
        class_embeddings,
        denoising_class_embed,
    ):
        # memory
        memory, mask_flatten, spatial_shapes, level_start_index, valid_ratios = (
            self.image_encoder(tensor_list)
        )

        # visual embeddings
        visual_embeds_list = []
        for reference_tensor_list in reference_tensor_list_list:
            reference_points = reference_tensor_list.tensors
            reference_points_mask = reference_tensor_list.mask
            visual_embed = self.visual_prompt_encoder(
                memory=memory,
                mask_flatten=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                reference_points_mask=reference_points_mask,
            )
            visual_embeds_list.append(visual_embed)

        # box decoder
        out_bboxes, out_logits, dn_out_bboxes, dn_out_logits, dn_meta = (
            self.box_decoder.forward_for_training(
                memory=memory,
                mask_flatten=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                visual_embed=None,
                targets=targets,
                num_classes=num_classes,
                class_embeddings=class_embeddings,
                denoising_class_embed=denoising_class_embed,
            )
        )
        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
        if dn_meta is not None:

            out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
            out["dn_meta"] = dn_meta
        return out, visual_embeds_list

    def forward(
        self,
        tensor_list: NestedTensor,
        reference_tensor_list: NestedTensor = None,
        targets=None,
    ):
        # memory
        memory, mask_flatten, spatial_shapes, level_start_index, valid_ratios = (
            self.image_encoder(tensor_list)
        )

        # visual embed
        visual_embed = None
        if reference_tensor_list is not None:
            reference_points = reference_tensor_list.tensors
            reference_points_mask = reference_tensor_list.mask
            visual_embed = self.visual_prompt_encoder(
                memory=memory,
                mask_flatten=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                reference_points_mask=reference_points_mask,
            )

        # box decoder
        out_bboxes, out_logits, dn_out_bboxes, dn_out_logits, dn_meta = (
            self.box_decoder(
                memory=memory,
                mask_flatten=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                visual_embed=visual_embed[:, -1:] if visual_embed is not None else None,
                targets=targets,
            )
        )
        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
        if dn_meta is not None:

            out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
            out["dn_meta"] = dn_meta
        return out, visual_embed

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    # new code
    def forward_for_training_model_without_ref_points(
        self,
        tensor_list: NestedTensor,
        targets,
        num_classes,
        class_embeddings,
        denoising_class_embed,
    ):
        # memory
        memory, mask_flatten, spatial_shapes, level_start_index, valid_ratios = (
            self.image_encoder(tensor_list)
        )

        # visual embeddings
        visual_embeds_list = []

        # box decoder
        out_bboxes, out_logits, dn_out_bboxes, dn_out_logits, dn_meta = (
            self.box_decoder.forward_for_training(
                memory=memory,
                mask_flatten=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                visual_embed=None,
                targets=targets,
                num_classes=num_classes,
                class_embeddings=class_embeddings,
                denoising_class_embed=denoising_class_embed,
            )
        )
        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
        if dn_meta is not None:
            out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
            out["dn_meta"] = dn_meta
        return out, visual_embeds_list

    def forward_for_training_model_with_ref_points_lst_lst(
        self,
        tensor_list: NestedTensor,
        reference_tensor_list_list: List[NestedTensor],
    ):
        # memory
        memory, mask_flatten, spatial_shapes, level_start_index, valid_ratios = (
            self.image_encoder(tensor_list)
        )

        # visual embeddings
        visual_embeds_list = []
        for reference_tensor_list in reference_tensor_list_list:
            reference_points = reference_tensor_list.tensors
            reference_points_mask = reference_tensor_list.mask
            visual_embed = self.visual_prompt_encoder(
                memory=memory,
                mask_flatten=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                reference_points_mask=reference_points_mask,
            )
            visual_embeds_list.append(visual_embed)

        # box decoder
        out_bboxes, out_logits, dn_out_bboxes, dn_out_logits, dn_meta = (
            self.box_decoder(
                memory=memory,
                mask_flatten=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                visual_embed=torch.cat(
                    [visual_embed[:, -1:] for visual_embed in visual_embeds_list], dim=1
                ),
                targets=None,
            )
        )
        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
        if dn_meta is not None:
            out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
            out["dn_meta"] = dn_meta
        return out, visual_embeds_list
