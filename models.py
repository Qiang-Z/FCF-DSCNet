"""DSCNet with a single bottleneck Fine-Curve adapter; validation pilot v1."""
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'snapshot/deps'), str(ROOT / 'snapshot'),
                str(ROOT / 'snapshot/backend'), str(ROOT / 'vendor/dscnet_source')]
from S3_DSCNet_pro import DSCNet_pro
from S3_DSConv_pro import DSConv_pro
from cracklab.modules.verified_v3 import FineVerifiedRouting
from cracklab.objective_v2 import ObjectiveV2

METHODS = ('dscnet', 'dscnet_aux', 'dscnet_fc_v2', 'dscnet_fc_v3_fixed')


class DSCFine(nn.Module):
    def __init__(self, method, seed):
        super().__init__()
        self.method = method
        # Backbone construction has exactly the same RNG state in all four arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = DSCNet_pro(n_channels=3, n_classes=1, kernel_size=9,
                                 extend_scope=1.0, if_offset=True,
                                 device='cpu', number=16, dim=1)
            self.net.sigmoid = nn.Identity()
        self._fine = None
        self._aux = {}
        if method != 'dscnet':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 1000)
                # conv3 is H/2 with 32 channels; evidence is H/4 with 16 channels.
                self.evidence_project = nn.Sequential(nn.AvgPool2d(2),
                                                      nn.Conv2d(32, 16, 1), nn.SiLU())
                mode = 'fine_curve_v3' if method == 'dscnet_fc_v3_fixed' else 'fine_curve_v2'
                self.feedback = FineVerifiedRouting(128, 16, mode, steps=4,
                                                     feedback_scale_init=0.05)
                # v3 transport and v2 residual both initially preserve DSCNet logits.
                nn.init.zeros_(self.feedback.output.weight)
                nn.init.zeros_(self.feedback.output.bias)
            self.net.conv3.register_forward_hook(self._capture_evidence)
            self.net.conv7.register_forward_hook(self._inject)

    def _capture_evidence(self, module, inputs, output):
        self._fine = self.evidence_project(output)

    def _inject(self, module, inputs, output):
        if self._fine is None:
            raise RuntimeError('Missing DSCNet shallow evidence')
        corrected, aux = self.feedback(output, self._fine)
        self._aux = {'verified_v3': aux}
        # Includes exactly the same four supervised heads, including centerline.
        return output if self.method == 'dscnet_aux' else corrected

    def forward(self, image, return_aux=False):
        self._fine, self._aux = None, {}
        for layer in self.net.modules():
            if isinstance(layer, DSConv_pro):
                layer.device = image.device
        try:
            logits = self.net(image)
            aux = self._aux
        finally:
            self._fine, self._aux = None, {}
        return (logits, aux) if return_aux else logits


def create_model(method, seed):
    if method not in METHODS:
        raise ValueError(method)
    return DSCFine(method, seed)


class Objective(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.base = ObjectiveV2(centerline_weight=0.15, path_weight=0.1,
                                junction_weight=0.05)

    def forward(self, logits, aux, target):
        return self.base(logits, aux, target)
