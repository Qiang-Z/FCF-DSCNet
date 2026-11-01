"""Fine-image edge verification and heading-preserving feature transport.

Predictions alone control inference. Native-mask corridor labels are training-only.
All four v3 ablations create identical heads, projections and transition parameters.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .verified import shift

DIRECTIONS = ((-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1))
OFFSETS_V3 = tuple((r*y,r*x) for r in (1,2) for y,x in DIRECTIONS)
REVERSE_V3 = tuple((k//8)*8+(k%8+4)%8 for k in range(16))


def transition_matrices(logits):
    d=logits.new_tensor(DIRECTIONS)
    d=F.normalize(d,dim=1); dot=d@d.T
    straight=logits.masked_fill(dot<0.70, -torch.inf).softmax(-1)
    branch=logits.masked_fill(dot<-0.99, -torch.inf).softmax(-1)
    return straight,branch


class FineVerifiedRouting(nn.Module):
    def __init__(self,channels,fine_channels=16,mode='fine_directional',steps=4,feedback_scale_init=None):
        super().__init__()
        if mode not in ('fine_aux','fine_curve','fine_curve_v2','fine_curve_v3','fine_pointwise','fine_routing','fine_directional'):
            raise ValueError(mode)
        self.mode,self.steps=mode,steps
        self.query=nn.Conv2d(channels,8,1)
        self.fine_project=nn.Conv2d(fine_channels,8,1)
        self.edge=nn.Sequential(nn.Conv2d(8*2+9*8+2,24,1),nn.SiLU(),nn.Conv2d(24,1,1))
        self.evidence=nn.Conv2d(channels+fine_channels,1,3,padding=1)
        self.junction=nn.Conv2d(channels+fine_channels,1,3,padding=1)
        # Curve-aware feedback is instantiated only for fine_curve so the
        # original fine_aux remains an unchanged auxiliary-only control.
        if mode in ('fine_curve','fine_curve_v2','fine_curve_v3'):
            self.centerline=nn.Conv2d(channels+fine_channels,1,3,padding=1)
            self.feedback=nn.Sequential(nn.Conv2d(channels*2,channels,1),nn.SiLU(),
                                        nn.Conv2d(channels,channels,1))
            nn.init.zeros_(self.feedback[-1].weight)
            nn.init.zeros_(self.feedback[-1].bias)
            # Keep the residual exactly zero at initialization via the last
            # layer, while leaving a nonzero scale so the branch can learn.
            default_feedback_scale=0.10 if mode=='fine_curve_v2' else (0.08 if mode=='fine_curve_v3' else 0.05)
            if feedback_scale_init is None:
                feedback_scale_init=default_feedback_scale
            self.feedback_scale=nn.Parameter(torch.tensor(float(feedback_scale_init)))
        self.value=nn.Conv2d(channels,channels,1)
        self.output=nn.Conv2d(channels,channels,1)
        self.scale=nn.Parameter(torch.tensor(0.1))
        self.transition=nn.Parameter(torch.eye(8)*2)

    def sampled_paths(self,fine,h,w):
        # align_corners=False: coarse cell centers and their 9 along-edge samples.
        yy,xx=torch.meshgrid(torch.arange(h,device=fine.device,dtype=fine.dtype),
                             torch.arange(w,device=fine.device,dtype=fine.dtype),indexing='ij')
        offsets=fine.new_tensor(OFFSETS_V3)
        t=torch.linspace(0,1,9,device=fine.device,dtype=fine.dtype)
        ys=yy[:,:,None,None]+offsets[None,None,:,0,None]*t
        xs=xx[:,:,None,None]+offsets[None,None,:,1,None]*t
        grid=torch.stack((2*(xs+0.5)/w-1,2*(ys+0.5)/h-1),-1)
        grid=grid.reshape(1,h,w*16*9,2).expand(fine.shape[0],-1,-1,-1)
        samples=F.grid_sample(fine,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
        return samples.reshape(fine.shape[0],8,h,w,16,9).permute(0,4,1,5,2,3).reshape(-1,72,h,w)

    def forward(self,x,fine):
        b,c,h,w=x.shape
        q=self.query(x)
        start=q[:,None].expand(-1,16,-1,-1,-1).reshape(b*16,8,h,w)
        end=torch.stack([shift(q,dy,dx) for dy,dx in OFFSETS_V3],1).reshape(b*16,8,h,w)
        geometry=x.new_tensor(OFFSETS_V3).div(2).view(1,16,2,1,1).expand(b,-1,-1,h,w).reshape(b*16,2,h,w)
        path=self.sampled_paths(self.fine_project(fine),h,w)
        edge_logits=self.edge(torch.cat((start,end,path,geometry),1)).reshape(b,16,h,w)
        context=torch.cat((x,F.adaptive_avg_pool2d(fine,(h,w))),1)
        evidence=self.evidence(context);junction=self.junction(context)
        p=edge_logits.sigmoid();ones=torch.ones_like(p[:,:1])
        gates=torch.cat([(p[:,k:k+1]*shift(p[:,REVERSE_V3[k]:REVERSE_V3[k]+1],dy,dx)+1e-8).sqrt()
                         *shift(ones,dy,dx) for k,(dy,dx) in enumerate(OFFSETS_V3)],1)
        aux=dict(edge_logits_v3=edge_logits,evidence_logits_v3=evidence,
                 junction_logits_v3=junction,gates_v3=gates)
        if self.mode=='fine_aux':return x,aux
        source=self.value(x)
        if self.mode in ('fine_curve','fine_curve_v2'):
            # Reciprocal corridor gates aggregate only along predicted local
            # curve directions. The centerline/evidence product suppresses
            # propagation in broad background regions and at uncertain edges.
            weights=gates/(1+gates.sum(1,keepdim=True))
            context_state=sum(weights[:,k:k+1]*shift(source,dy,dx)
                              for k,(dy,dx) in enumerate(OFFSETS_V3))
            centerline=self.centerline(context)
            aux['centerline_logits_v3']=centerline
            quality_raw=(centerline.sigmoid()*evidence.sigmoid()).clamp(0,1)
            residual=self.feedback(torch.cat((source,context_state),1))
            if self.mode=='fine_curve_v2':
                # Preserve a conservative identity path in low-contrast regions:
                # v1's product gate could suppress useful propagation entirely.
                quality=0.35+0.65*quality_raw
                alpha=0.2*torch.tanh(self.feedback_scale)
            else:
                quality=quality_raw
                alpha=0.1*torch.tanh(self.feedback_scale)
            return x+alpha*quality*residual,aux
        if self.mode=='fine_curve_v3':
            # v3 reuses the directional transport below as its main path;
            # this head supplies only the topology-aware residual gate.
            centerline=self.centerline(context)
            aux['centerline_logits_v3']=centerline
            quality_raw=(centerline.sigmoid()*evidence.sigmoid()).clamp(0,1)
        if self.mode=='fine_pointwise':
            state=source
        elif self.mode=='fine_routing':
            weights=gates/(1+gates.sum(1,keepdim=True))
            restart=1-weights.sum(1,keepdim=True);state=source
            for _ in range(self.steps):
                state=restart*source+sum(weights[:,k:k+1]*shift(state,dy,dx)
                                         for k,(dy,dx) in enumerate(OFFSETS_V3))
        else:
            # Separate state for each look-ahead heading. Only a predicted junction
            # opens large heading changes; immediate reverse transitions are masked.
            straight,branch=transition_matrices(self.transition)
            state=source[:,None].expand(-1,8,-1,-1,-1)
            g=gates.reshape(b,2,8,h,w);mass=g.sum(1)
            weights=g/(1+mass[:,None]);restart=1-weights.sum(1)
            j=junction.sigmoid()[:,None]
            for _ in range(self.steps):
                sm=torch.einsum('dq,bqchw->bdchw',straight,state)
                bm=torch.einsum('dq,bqchw->bdchw',branch,state)
                mixed=sm*(1-j)+bm*j
                moved=[]
                for d in range(8):
                    value=restart[:,d:d+1]*source
                    for radius in range(2):
                        dy,dx=OFFSETS_V3[radius*8+d]
                        value=value+weights[:,radius,d:d+1]*shift(mixed[:,d],dy,dx)
                    moved.append(value)
                state=torch.stack(moved,1)
            readout=(1+mass)/(8+mass.sum(1,keepdim=True))
            state=(state*readout[:,:,None]).sum(1)
            if self.mode=='fine_curve_v3':
                # v3 keeps the stronger fine_directional transport as the
                # primary path, then adds a zero-initialized, centerline-
                # gated residual. This preserves the directional baseline
                # while allowing topology-aware correction on uncertain bends.
                curve_context=sum(weights[:,radius,d:d+1]*shift(source,dy,dx)
                                  for radius in range(2) for d in range(8)
                                  for dy,dx in (OFFSETS_V3[radius*8+d],))
                quality=0.35+0.65*quality_raw
                residual=self.feedback(torch.cat((source,curve_context),1))
                alpha=0.2*torch.tanh(self.feedback_scale)
                return x+self.scale*self.output(state)+alpha*quality*residual,aux
        return x+self.scale*self.output(state),aux
