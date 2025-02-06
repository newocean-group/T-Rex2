from typing import Optional
from torch import Tensor
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from packaging import version
import torchvision
from typing import Optional, List
from torchvision import ops
import torch.nn.functional as F

if version.parse(torchvision.__version__) < version.parse("0.7"):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


def read_classes_name(file_path):
    names = {}
    with open(file_path, "r") as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip("\n")
    return names


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
        if mask == "auto":
            self.mask = torch.zeros_like(tensors).to(tensors.device)
            if self.mask.dim() == 3:
                self.mask = self.mask.sum(0).to(bool)
            elif self.mask.dim() == 4:
                self.mask = self.mask.sum(1).to(bool)
            else:
                raise ValueError(
                    "tensors dim must be 3 or 4 but {}({})".format(
                        self.tensors.dim(), self.tensors.shape
                    )
                )


def get_new_checkpoint(checkpoint1, checkpoint2):
    new_checkpoints = {}
    count = 0
    keys2 = list(checkpoint2.keys())
    for key1 in checkpoint1.keys():
        if "image_encoder.backbone" in key1:
            found = False
            k = ".".join(key1.split(".")[2:])
            for key2 in keys2:
                if k in key2:
                    if checkpoint1[key1].shape != checkpoint2[key2].shape:
                        new_checkpoints[key1] = checkpoint1[key1]
                    else:
                        new_checkpoints[key1] = checkpoint2[key2]
                        count += 1
                    found = True
                    break
            if not found:
                new_checkpoints[key1] = checkpoint1[key1]
        else:
            new_checkpoints[key1] = checkpoint1[key1]
    print("count value: ", count)
    return new_checkpoints


# Colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]
COLORS *= 900

revert_normalization = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def plot_im_with_boxes(
    im, boxes, probs=None, ax=None, draw_text=False, classes_map=None
):
    if ax is None:
        plt.imshow(im)
        ax = plt.gca()

    for i, b in enumerate(boxes.tolist()):
        xmin, ymin, xmax, ymax = b

        patch = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            color=COLORS[i],
            linewidth=2,
        )

        ax.add_patch(patch)

        if draw_text:
            if probs is not None:
                if probs.ndim == 1:
                    cl = probs[i].item()
                    text = f"{classes_map[cl]}"
                else:
                    cl = probs[i].argmax().item()
                    text = f"{classes_map[cl]}: {probs[i,cl]:0.2f}"
            else:
                text = ""

            ax.text(
                xmin, ymin, text, fontsize=7, bbox=dict(facecolor="yellow", alpha=0.5)
            )


def inverse_sigmoid(x, eps=1e-8):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse("0.7"):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(
            input, size, scale_factor, mode, align_corners
        )


def draw_result(
    tensor_list, ori_shapes, target, model, classes_map, threshold=0.25, idx=0
):
    model.eval()
    outs, _ = model(tensor_list)
    final_cls_outs = outs["pred_logits"].sigmoid().cpu()[idx]
    final_cls_outs_value, final_cls_outs_idx = final_cls_outs.max(dim=-1)
    final_bbox_outs = outs["pred_boxes"].cpu()[idx]

    o_keep = final_cls_outs_value >= threshold
    final_cls_outs_idx = final_cls_outs_idx[o_keep]
    final_bbox_outs = final_bbox_outs[o_keep]

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    ori_h, ori_w = ori_shapes[idx]
    ori_img = tensor_list.tensors[idx][:, :ori_h, :ori_w]
    im = revert_normalization(ori_img).permute(1, 2, 0).cpu().clip(0, 1)
    h, w, _ = im.shape

    t_cl = target[idx]["labels"].clone()
    t_bbox = target[idx]["boxes"].clone()

    t_bbox[:, [0, 2]] = t_bbox[:, [0, 2]] * w
    t_bbox[:, [1, 3]] = t_bbox[:, [1, 3]] * h
    t_bbox = ops.box_convert(t_bbox, in_fmt="cxcywh", out_fmt="xyxy")

    final_bbox_outs[:, [0, 2]] = final_bbox_outs[:, [0, 2]] * w
    final_bbox_outs[:, [1, 3]] = final_bbox_outs[:, [1, 3]] * h
    final_bbox_outs = ops.box_convert(final_bbox_outs, in_fmt="cxcywh", out_fmt="xyxy")

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(im)
    plot_im_with_boxes(im, t_bbox, t_cl, ax, draw_text=True, classes_map=classes_map)
    ax.set_axis_off()

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(im)
    plot_im_with_boxes(
        im,
        final_bbox_outs,
        final_cls_outs_idx,
        ax,
        draw_text=True,
        classes_map=classes_map,
    )
    ax.set_axis_off()

    fig.savefig("./results.jpg", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)  # Close figure to release memory


def draw_result_for_visual_prompt(
    tensor_list, reference_tensor_list, ori_shapes, target, model, threshold=0.25, idx=0
):
    model.eval()
    outs, _ = model(tensor_list, reference_tensor_list)
    final_cls_outs = outs["pred_logits"].sigmoid().cpu()[idx].view(-1)
    final_bbox_outs = outs["pred_boxes"].cpu()[idx]

    o_keep = final_cls_outs >= threshold
    final_cls_outs = final_cls_outs[o_keep]
    final_bbox_outs = final_bbox_outs[o_keep]

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    ori_h, ori_w = ori_shapes[idx]
    ori_img = tensor_list.tensors[idx][:, :ori_h, :ori_w]
    im = revert_normalization(ori_img).permute(1, 2, 0).cpu().clip(0, 1)
    h, w, _ = im.shape

    t_cl = target[idx]["labels"].clone()
    t_bbox = target[idx]["boxes"].clone()

    t_bbox[:, [0, 2]] = t_bbox[:, [0, 2]] * w
    t_bbox[:, [1, 3]] = t_bbox[:, [1, 3]] * h
    t_bbox = ops.box_convert(t_bbox, in_fmt="cxcywh", out_fmt="xyxy")

    ref_bbox = reference_tensor_list.tensors[idx]
    ref_bbox_mask = reference_tensor_list.mask[idx]
    ref_bbox = ref_bbox[~ref_bbox_mask]

    ref_bbox[:, [0, 2]] = ref_bbox[:, [0, 2]] * w
    ref_bbox[:, [1, 3]] = ref_bbox[:, [1, 3]] * h
    ref_bbox = ops.box_convert(ref_bbox, in_fmt="cxcywh", out_fmt="xyxy")

    final_bbox_outs[:, [0, 2]] = final_bbox_outs[:, [0, 2]] * w
    final_bbox_outs[:, [1, 3]] = final_bbox_outs[:, [1, 3]] * h
    final_bbox_outs = ops.box_convert(final_bbox_outs, in_fmt="cxcywh", out_fmt="xyxy")

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(im)
    plot_im_with_boxes(im, ref_bbox, torch.tensor([1] * ref_bbox.shape[0]), ax)
    ax.set_axis_off()

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(im)
    plot_im_with_boxes(im, t_bbox, t_cl, ax)
    ax.set_axis_off()

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(im)
    plot_im_with_boxes(
        im, final_bbox_outs, torch.tensor([1] * final_cls_outs.shape[0]), ax
    )
    ax.set_axis_off()

    fig.savefig(
        "./results_with_visual_prompts.jpg", bbox_inches="tight", pad_inches=0.1
    )
    plt.close(fig)  # Close figure to release memory


def draw_result_for_fsc(
    tensor_list, reference_tensor_list, ori_shapes, model, threshold=0.25, idx=0
):
    model.eval()
    outs, _ = model(tensor_list, reference_tensor_list)
    final_cls_outs = outs["pred_logits"].sigmoid().cpu()[idx].view(-1)
    final_bbox_outs = outs["pred_boxes"].cpu()[idx]

    o_keep = final_cls_outs >= threshold
    final_cls_outs = final_cls_outs[o_keep]
    final_bbox_outs = final_bbox_outs[o_keep]

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    ori_h, ori_w = ori_shapes[idx]
    ori_img = tensor_list.tensors[idx][:, :ori_h, :ori_w]
    im = revert_normalization(ori_img).permute(1, 2, 0).cpu().clip(0, 1)
    h, w, _ = im.shape

    ref_bbox = reference_tensor_list.tensors[idx]
    ref_bbox_mask = reference_tensor_list.mask[idx]
    ref_bbox = ref_bbox[~ref_bbox_mask]

    ref_bbox[:, [0, 2]] = ref_bbox[:, [0, 2]] * w
    ref_bbox[:, [1, 3]] = ref_bbox[:, [1, 3]] * h
    ref_bbox = ops.box_convert(ref_bbox, in_fmt="cxcywh", out_fmt="xyxy")

    final_bbox_outs[:, [0, 2]] = final_bbox_outs[:, [0, 2]] * w
    final_bbox_outs[:, [1, 3]] = final_bbox_outs[:, [1, 3]] * h
    final_bbox_outs = ops.box_convert(final_bbox_outs, in_fmt="cxcywh", out_fmt="xyxy")

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(im)
    plot_im_with_boxes(im, ref_bbox, torch.tensor([1] * ref_bbox.shape[0]), ax)
    ax.set_axis_off()

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(im)
    plot_im_with_boxes(
        im, final_bbox_outs, torch.tensor([1] * final_cls_outs.shape[0]), ax
    )
    ax.set_axis_off()

    fig.savefig("./results_for_fsc.jpg", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)  # Close figure to release memory
