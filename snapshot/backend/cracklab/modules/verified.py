"""Sparse path transport with reciprocal endpoint and along-path evidence."""
import torch
from torch import nn
from torch.nn import functional as F


OFFSETS = ((-2,0),(0,2),(2,0),(0,-2),(-2,2),(2,2),(2,-2),(-2,-2))
OPPOSITE = (2,3,0,1,6,7,4,5)


def shift(x, dy, dx):
    # Read x at destination + offset; outside values are exactly zero.
    h,w=x.shape[-2:]; p=max(abs(dy),abs(dx))
    z=F.pad(x,(p,p,p,p))
    return z[...,p+dy:p+dy+h,p+dx:p+dx+w]


def reciprocal_gates(endpoint, evidence, reciprocal=True, path_evidence=True):
    gates=[]
    ones=torch.ones_like(evidence)
    for k,(dy,dx) in enumerate(OFFSETS):
        valid=shift(ones,dy,dx)
        g=endpoint[:,k:k+1]
        if reciprocal:
            reverse=shift(endpoint[:,OPPOSITE[k]:OPPOSITE[k]+1],dy,dx)
            g=(g*reverse+1e-8).sqrt()
        if path_evidence:
            # Three samples, including midpoint; soft support avoids a hard veto
            # on weak true gaps. Supervision disambiguates false bridges.
            g=g*(evidence+shift(evidence,dy//2,dx//2)+shift(evidence,dy,dx))/3
        gates.append(g*valid)
    return torch.cat(gates,1)


class VerifiedPathRouting(nn.Module):
    def __init__(self, channels, steps=4, reciprocal=True, path_evidence=True):
        super().__init__()
        self.steps,self.reciprocal,self.path_evidence=steps,reciprocal,path_evidence
        self.endpoint=nn.Conv2d(channels,8,1)
        self.evidence=nn.Conv2d(channels,1,3,padding=1)
        self.value=nn.Conv2d(channels,channels,1)
        self.output=nn.Conv2d(channels,channels,1)
        self.scale=nn.Parameter(torch.tensor(0.1))

    def forward(self,x):
        endpoint=self.endpoint(x); evidence=self.evidence(x)
        gates=reciprocal_gates(endpoint.sigmoid(),evidence.sigmoid(),self.reciprocal,self.path_evidence)
        weights=gates/(1+gates.sum(1,keepdim=True))
        source=self.value(x); state=source
        for _ in range(self.steps):
            transported=sum(weights[:,k:k+1]*shift(state,dy,dx) for k,(dy,dx) in enumerate(OFFSETS))
            state=transported+(1-weights.sum(1,keepdim=True))*source
        return x+self.scale*self.output(state),dict(endpoint_logits=endpoint,evidence_logits=evidence,gates=gates)
