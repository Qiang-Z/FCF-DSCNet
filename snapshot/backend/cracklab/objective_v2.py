"""Training-only structural targets and measured-gain selector supervision."""
import numpy as np
from scipy.ndimage import label
import torch
from torch import nn
from torch.nn import functional as F
from .losses import SegmentationObjective, balanced_binary_mean
from .modules.common import patchify
from .modules.verified import OFFSETS, shift
from .corridor_targets import batch_corridor_targets,batch_centerline_targets


def tiled_component_labels(mask, patch):
    """Label all tiles in one CPU pass with zero gutters; paths cannot leave tiles."""
    h,w=mask.shape
    if h%patch or w%patch: raise ValueError('Target shape not divisible by patch')
    tiles=mask.reshape(h//patch,patch,w//patch,patch).transpose(0,2,1,3)
    separated=np.pad(tiles,((0,0),(0,0),(0,1),(0,1)))
    separated=separated.transpose(0,2,1,3).reshape(h//patch*(patch+1),w//patch*(patch+1))
    labels=label(separated,np.ones((3,3)))[0]
    return labels.reshape(h//patch,patch+1,w//patch,patch+1).transpose(0,2,1,3)[...,:patch,:patch].reshape(h//patch,w//patch,patch*patch)


def pair_targets(target,patch,indices):
    idx=indices.detach().cpu().numpy()
    masks=target.detach().cpu().numpy()[:,0]>.5
    labels=np.stack([tiled_component_labels(m,patch)[...,idx] for m in masks])
    same=(labels[..., :,None]==labels[...,None,:]) & (labels[..., :,None]>0)
    return torch.as_tensor(same,device=target.device),torch.as_tensor(labels>0,device=target.device)


def local_topology_error(pred,target):
    """Patch-local split/merge diagnostic at a fixed 8x8 anchor grid.

    This is not a homotopy guarantee or a published topology metric. Also
    includes foreground misses and extra pixels when no anchor hits a crack.
    """
    lp=label(pred,np.ones((3,3)))[0]; lt=label(target,np.ones((3,3)))[0]
    ys=np.linspace(0,pred.shape[0]-1,8).astype(int)
    xs=np.linspace(0,pred.shape[1]-1,8).astype(int)
    # Snap probes to target foreground inside each small grid cell by max label.
    def probes(labels):
        out=[]
        for y in ys:
            for x in xs:
                out.append(labels[max(0,y-1):y+2,max(0,x-1):x+2].max())
        return np.asarray(out)
    a,b=probes(lp),probes(lt)
    pa=(a[:,None]==a[None,:])&(a[:,None]>0)
    pb=(b[:,None]==b[None,:])&(b[:,None]>0)
    valid=~np.eye(64,dtype=bool)
    pos=pb&valid; neg=~pb&valid
    miss=float((~pa&pos).sum()/max(1,pos.sum()))
    merge=float((pa&neg).sum()/max(1,neg.sum()))
    pixel=float(np.logical_xor(pred,target).mean())
    return .4*miss+.4*merge+.2*pixel


class ObjectiveV2(nn.Module):
    def __init__(self, port_weight=.1, consistency_weight=.05, path_weight=.1,
                 selector_weight=.1, junction_weight=.05, centerline_weight=.15, **unused):
        super().__init__()
        self.base=SegmentationObjective(slot_weight=0,curve_weight=0)
        self.pw,self.cw,self.vw,self.sw=port_weight,consistency_weight,path_weight,selector_weight
        self.jw,self.clw=junction_weight,centerline_weight

    def forward(self,logits,aux,target):
        total,parts=self.base(logits,{},target)
        if 'ports' in aux:
            c=aux['ports']
            truth=F.interpolate(target,c['size'],mode='nearest')
            for name,n,idx,key in [('child',4,c['indices'],'child_logits'),('parent',8,c['parent_indices'],'parent_logits')]:
                pair,fg=pair_targets(truth,n,idx)
                valid=~torch.eye(len(idx),device=truth.device,dtype=torch.bool)
                cost=F.binary_cross_entropy_with_logits(c[key],pair.float(),reduction='none')
                term=balanced_binary_mean(cost,pair,valid.expand_as(pair))
                parts['port_'+name]=term
                total=total+self.pw*term
                if name=='child':
                    occ=F.binary_cross_entropy_with_logits(c['occupancy_logits'],fg.float(),reduction='none')
                    term=balanced_binary_mean(occ,fg,torch.ones_like(fg))
                    parts['port_occupancy']=term; total=total+self.pw*term
            # Compare off-diagonal row distributions. Mask diagonal self-return
            # paths so loops do not dominate the cross-scale consistency loss.
            n=c['direct'].shape[-1]; eye=torch.eye(n,device=truth.device)
            a=c['direct']*(1-eye); b=c['composed']*(1-eye)
            a=a/(a.sum(-1,keepdim=True)+1e-6); b=b/(b.sum(-1,keepdim=True)+1e-6)
            term=(a-b).square().mean()
            parts['port_consistency']=term; total=total+self.cw*term
        if 'verified' in aux:
            c=aux['verified']; shape=c['evidence_logits'].shape[-2:]
            # Max-pooling retains thin traces; any positive coarse cell is evidence.
            truth=F.adaptive_max_pool2d(target,shape)
            valid=torch.ones_like(truth)
            targets=[]; masks=[]
            for dy,dx in OFFSETS:
                targets.append(truth*shift(truth,dy//2,dx//2)*shift(truth,dy,dx))
                masks.append(shift(valid,dy,dx))
            targets=torch.cat(targets,1); masks=torch.cat(masks,1)>0
            costs=F.binary_cross_entropy_with_logits(c['endpoint_logits'],targets,reduction='none')
            edge=balanced_binary_mean(costs,targets>0,masks)
            costs=F.binary_cross_entropy_with_logits(c['evidence_logits'],truth,reduction='none')
            ev=balanced_binary_mean(costs,truth>0,valid>0)
            parts.update(path_edges=edge,path_evidence=ev)
            total=total+self.vw*(edge+ev)
        if 'verified_v3' in aux:
            c=aux['verified_v3']
            edges,valid,occupancy,junction=batch_corridor_targets(target,c['edge_logits_v3'].shape[-2:])
            for name,logits,truth,mask,weight in (
                ('corridor_edges',c['edge_logits_v3'],edges,valid,self.vw),
                ('fine_evidence',c['evidence_logits_v3'],occupancy,torch.ones_like(occupancy,dtype=torch.bool),self.vw),
                ('junction',c['junction_logits_v3'],junction,torch.ones_like(junction,dtype=torch.bool),self.jw)):
                cost=F.binary_cross_entropy_with_logits(logits.float(),truth.float(),reduction='none')
                term=balanced_binary_mean(cost,truth>0,mask)
                parts[name]=term;total=total+weight*term
            if 'centerline_logits_v3' in c:
                centerline=batch_centerline_targets(target,c['centerline_logits_v3'].shape[-2:])
                cost=F.binary_cross_entropy_with_logits(c['centerline_logits_v3'].float(),centerline.float(),reduction='none')
                term=balanced_binary_mean(cost,centerline>0,torch.ones_like(centerline,dtype=torch.bool))
                parts['centerline']=term;total=total+self.clw*term
        if 'refinement' in aux and aux['refinement']['mode']=='learned':
            c=aux['refinement']; patch=c['patch']; ids=c['indices']
            tiles,_=patchify(target,patch); tiles=tiles.flatten(1,2)
            selected=tiles.gather(1,ids[:,:,None,None].expand(-1,-1,patch*patch,1))
            selected=selected.reshape(-1,1,patch,patch)
            # Supervise actual improvement on executed regions, not GT at inference.
            before=c['before'].detach(); after=c['after'].detach()
            pixel_gain=(F.binary_cross_entropy_with_logits(before,selected,reduction='none')-
                        F.binary_cross_entropy_with_logits(after,selected,reduction='none')).mean((1,2,3))
            bp=(before.sigmoid()>=.5).cpu().numpy()[:,0]
            ap=(after.sigmoid()>=.5).cpu().numpy()[:,0]
            tt=(selected>.5).cpu().numpy()[:,0]
            gains=[local_topology_error(b,t)-local_topology_error(a,t) for a,b,t in zip(ap,bp,tt)]
            desired=torch.tensor(gains,device=logits.device)+.2*pixel_gain
            predicted=c['scores'].gather(1,ids).flatten()
            term=F.smooth_l1_loss(predicted,desired)
            parts['selector_gain']=term; total=total+self.sw*term
        parts['total']=total
        return total,parts
