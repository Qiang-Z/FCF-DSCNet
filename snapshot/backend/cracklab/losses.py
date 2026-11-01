"""Training-only mask-derived targets; inference never receives ground truth."""
import torch
from torch import nn
from torch.nn import functional as F
from .modules.common import patchify, masked_mean


def balanced_binary_mean(values, positive, valid):
    positive, valid = positive.bool(), valid.bool()
    pos, neg = positive & valid, (~positive) & valid
    # Empty classes contribute neither a spurious zero nor a NaN.
    present = (pos.any().float() + neg.any().float()).clamp_min(1)
    return (masked_mean(values, pos) + masked_mean(values, neg)) / present


@torch.no_grad()
def component_pairs(foreground, neighbor):
    """Exact 8-connected reachability INSIDE each patch, not global components."""
    fg = foreground.bool()
    n = fg.shape[-1]
    pairs = fg.unsqueeze(-1) & fg.unsqueeze(-2)
    reach = pairs & (neighbor | torch.eye(n, device=fg.device, dtype=torch.bool))
    # Boolean Floyd-Warshall is deterministic, with no CPU labeling round trip.
    for k in range(n):
        reach = reach | (reach[..., :, k:k+1] & reach[..., k:k+1, :])
    return reach


def slot_losses(cache, target, padded_size):
    h, w = target.shape[-2:]
    pad = (0, padded_size[1]-w, 0, padded_size[0]-h)
    mask, _ = patchify(F.pad(target, pad), cache["patch"])
    valid, _ = patchify(F.pad(torch.ones_like(target), pad), cache["patch"])
    fg, valid = mask[..., 0] > 0.5, valid[..., 0] > 0.5
    s = cache["assignments"].float()
    foreground = s[..., 1:].sum(-1).clamp(1e-6, 1-1e-6)
    binary = F.binary_cross_entropy(foreground, fg.float(), reduction="none")
    fg_loss = balanced_binary_mean(binary, fg, valid)
    pairs_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    neighbor = cache["neighbor"]
    adjacent_fg = fg.unsqueeze(-1) & fg.unsqueeze(-2)
    edge = F.binary_cross_entropy_with_logits(cache["affinity_logits"].float(), adjacent_fg.float(), reduction="none")
    edge_loss = balanced_binary_mean(edge, adjacent_fg, pairs_valid & neighbor)
    same = component_pairs(fg, neighbor)
    conditional = F.normalize(s[..., 1:], p=2, dim=-1, eps=1e-6)
    similarity = (conditional @ conditional.transpose(-1, -2)).clamp(0, 1)
    diagonal = torch.eye(fg.shape[-1], device=fg.device, dtype=torch.bool)
    pair_valid = pairs_valid & adjacent_fg & ~diagonal
    pair_loss = balanced_binary_mean((similarity-same.float()).square(), same, pair_valid)
    return {"slot_foreground": fg_loss, "slot_affinity": edge_loss, "slot_components": pair_loss}


def curve_losses(cache, target, padded_size, max_targets=512):
    """Point-set attraction supplements local raster gradients outside support.

    Targets are foreground pixels from masks, NOT annotated centerlines. Thus
    this is a geometry initialization aid, not exact skeleton supervision.
    """
    points, presence = cache["points"].float(), cache["presence"].float()
    h, w = padded_size
    scale = points.new_tensor([max(w-1, 1), max(h-1, 1)]) / 2
    predicted = (points+1)*scale / max(h, w)
    coverage, precision, empty, presence_targets = [], [], [], []
    for i in range(target.shape[0]):
        coords = torch.nonzero(target[i, 0] > 0.5, as_tuple=False)
        if not len(coords):
            empty.append(presence[i].mean())
            presence_targets.append(torch.tensor(0., device=presence.device))
            continue
        presence_targets.append(torch.tensor(1., device=presence.device))
        if len(coords) > max_targets:
            select = torch.linspace(0, len(coords)-1, max_targets, device=coords.device).long()
            coords = coords[select]
        truth = coords.flip(-1).float()/max(h, w)
        distance = torch.cdist(truth[None], predicted[i].flatten(0, 1)[None]).squeeze(0)
        coverage.append(distance.min(-1).values.mean())
        nearest = distance.min(0).values.reshape_as(points[i, ..., 0])
        # All queries receive a location gradient, even if their presence is low.
        precision.append((nearest.mean(-1)*(0.25+0.75*presence[i])).mean())
    zero = points.sum()*0
    return {
        "curve_coverage": torch.stack(coverage).mean() if coverage else zero,
        "curve_precision": torch.stack(precision).mean() if precision else zero,
        "curve_empty": torch.stack(empty).mean() if empty else zero,
        "curve_presence": F.binary_cross_entropy(
            presence.clamp(1e-5, 1-1e-5), torch.stack(presence_targets)[:, None].expand_as(presence)
        ) if presence_targets else zero,
    }


class SegmentationObjective(nn.Module):
    def __init__(self, slot_weight=0.2, curve_weight=5.0):
        super().__init__()
        self.slot_weight, self.curve_weight = slot_weight, curve_weight

    def forward(self, logits, aux, target):
        logits, target = logits.float(), target.float()
        prob = logits.sigmoid()
        intersection = (prob*target).sum((1, 2, 3))
        dice = 1-((2*intersection+1)/(prob.sum((1, 2, 3))+target.sum((1, 2, 3))+1)).mean()
        parts = {"bce": F.binary_cross_entropy_with_logits(logits, target), "dice_loss": dice}
        total = parts["bce"]+dice
        if "slots" in aux and self.slot_weight:
            terms = slot_losses(aux["slots"], target, aux["padded_size"])
            total = total+self.slot_weight*sum(terms.values())
            parts.update(terms)
        if "curve" in aux and self.curve_weight:
            terms = curve_losses(aux["curve"], target, aux["padded_size"])
            total = total+self.curve_weight*(terms["curve_coverage"]+0.25*terms["curve_precision"]+0.5*terms["curve_presence"])+terms["curve_empty"]
            parts.update(terms)
        parts["total"] = total
        return total, parts
