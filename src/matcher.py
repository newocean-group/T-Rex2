import torch
from torch import nn
from torchvision import ops
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    def __init__(
        self, cost_class: float = 2, cost_bbox: float = 5, cost_giou: float = 2
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(
                0, 1
            )  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (
                (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -ops.generalized_box_iou(
                ops.box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy"),
                ops.box_convert(tgt_bbox, in_fmt="cxcywh", out_fmt="xyxy"),
            )

            # Final cost matrix
            C = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [
                linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
            ]
            return [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in indices
            ]
