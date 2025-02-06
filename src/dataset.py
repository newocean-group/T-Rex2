from pathlib import Path
import torch
import torch.utils.data
from pycocotools import mask as coco_mask
import numpy as np
import src.transforms as T
from src.utils import NestedTensor, read_classes_name
from torchvision.datasets import CocoDetection


def pad_and_stack_with_mask(inputs):
    # Get max height and width
    max_height = max(tensor.shape[-2] for tensor in inputs)
    max_width = max(tensor.shape[-1] for tensor in inputs)

    ori_shapes = []

    padded_inputs = []
    input_masks = []

    for input in inputs:
        height, width = input.shape[-2], input.shape[-1]
        ori_shapes.append(torch.tensor([height, width], dtype=torch.int32))

        padded_input = torch.zeros(size=(3, max_height, max_width))
        padded_input[:, :height, :width] = input
        padded_inputs.append(padded_input)

        mask = torch.ones(size=(max_height, max_width))
        mask[:height, :width] = 0.0

        input_masks.append(mask.bool())

    # Stack padded tensors and masks
    new_inputs = torch.stack(padded_inputs)
    input_masks = torch.stack(input_masks)
    ori_shapes = torch.stack(ori_shapes)

    return new_inputs, input_masks, ori_shapes


def pad_and_stack_with_mask_for_visual_prompts(boxes):
    n_max_boxes = max([box.shape[0] for box in boxes])
    padded_boxes = []
    boxes_mask = []
    for box in boxes:
        padded_box = torch.zeros(size=(n_max_boxes, 4), dtype=torch.float32)
        n_box = box.shape[0]
        padded_box[:n_box] = box
        padded_boxes.append(padded_box)
        mask = torch.ones(size=(n_max_boxes,))
        mask[:n_box] = 0.0
        boxes_mask.append(mask.bool())
    padded_boxes = torch.stack(padded_boxes)
    boxes_mask = torch.stack(boxes_mask)
    return padded_boxes, boxes_mask


class CustomDataset(CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks=False,
    ):
        super(CustomDataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def set_classes_map(self, classes_map):
        self.classes_map = {}
        for k, v in classes_map.items():
            v = v["name"].lower()
            if "-" in v:
                v = " ".join(v.split("-"))
            elif "_" in v:
                v = " ".join(v.split("_"))
            else:
                v = v
            self.classes_map[k] = v
        print("classes map: ", self.classes_map)

    def __getitem__(self, idx):
        while True:
            try:
                img, target = super(CustomDataset, self).__getitem__(idx)
                image_id = self.ids[idx]
                target = {"image_id": image_id, "annotations": target}
                img, target = self.prepare(img, target)
                if self._transforms is not None:
                    img, target = self._transforms(img, target)
                classes = target["labels"]
                boxes = target["boxes"]
                if len(boxes) > 0:
                    break
                else:
                    idx = np.random.choice(len(self))
            except:
                idx = np.random.choice(len(self))
        target["text_list"] = [self.classes_map[v.item()] for v in classes]
        # target["labels"] = classes
        target["boxes"] = boxes.float()
        return img, target


def collate_fn(inputs):
    inputs_ = [i[0] for i in inputs]
    ori_shapes = [i[0].shape[1:] for i in inputs]
    text_list = [i[1]["text_list"] for i in inputs]
    boxes_list = [i[1]["boxes"] for i in inputs]
    return inputs_, boxes_list, text_list, ori_shapes


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build_custom_dataset(
    img_folder,
    ann_file,
    image_set="train",
):
    dataset = CustomDataset(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set),
        return_masks=False,
    )
    return dataset
