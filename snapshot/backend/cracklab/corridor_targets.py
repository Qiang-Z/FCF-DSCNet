"""Exact native-mask reachability inside finite candidate corridors.

No global component shortcuts, dilation bridges, or GT-dependent inference.
Ambiguous endpoint cells containing multiple corridor components are ignored.
"""
from functools import lru_cache
import numpy as np
from scipy.ndimage import label,maximum_filter
import torch
from .metrics import skeletonize
from .modules.verified_v3 import OFFSETS_V3,REVERSE_V3


@lru_cache(maxsize=64)
def corridor_geometry(dy,dx,sy,sx):
    hh,ww=(abs(dy)+1)*sy,(abs(dx)+1)*sx
    ay,ax=max(0,-dy)*sy,max(0,-dx)*sx
    by,bx=ay+dy*sy,ax+dx*sx
    y,x=np.mgrid[:hh,:ww]
    vy,vx=dy*sy,dx*sx
    cy,cx=ay+(sy-1)/2,ax+(sx-1)/2
    t=np.clip(((y-cy)*vy+(x-cx)*vx)/(vy*vy+vx*vx),0,1)
    corridor=(y-cy-t*vy)**2+(x-cx-t*vx)**2 <= (0.75*max(sy,sx))**2
    return corridor,(ay,ax),(by,bx)


def native_corridor_targets(mask,shape):
    mask=np.asarray(mask,dtype=bool);hh,ww=mask.shape;h,w=shape
    if hh%h or ww%w:raise ValueError('Targets must align to native coarse cells')
    sy,sx=hh//h,ww//w
    occupancy=mask.reshape(h,sy,w,sx).any((1,3))
    target=np.zeros((16,h,w),np.float32);valid=np.zeros_like(target,dtype=bool)
    for k,(dy,dx) in enumerate(OFFSETS_V3):
        if dy<0 or (dy==0 and dx<0):continue
        reverse=REVERSE_V3[k]
        ys,xs=np.mgrid[max(0,-dy):min(h,h-dy),max(0,-dx):min(w,w-dx)]
        valid[k,ys,xs]=True;valid[reverse,ys+dy,xs+dx]=True
        candidate=occupancy[ys,xs]&occupancy[ys+dy,xs+dx]
        corridor,(ay,ax),(by,bx)=corridor_geometry(dy,dx,sy,sx)
        for y,x in zip(ys[candidate],xs[candidate]):
            top,left=min(y,y+dy)*sy,min(x,x+dx)*sx
            patch=mask[top:top+corridor.shape[0],left:left+corridor.shape[1]] & corridor
            components=label(patch,np.ones((3,3)))[0]
            a=np.unique(components[ay:ay+sy,ax:ax+sx]);a=a[a>0]
            b=np.unique(components[by:by+sy,bx:bx+sx]);b=b[b>0]
            if len(a)>1 or len(b)>1:
                valid[k,y,x]=False;valid[reverse,y+dy,x+dx]=False
            else:
                connected=float(len(a)==1 and len(b)==1 and a[0]==b[0])
                target[k,y,x]=connected;target[reverse,y+dy,x+dx]=connected
    sk=skeletonize(mask)
    z=np.pad(sk,1)
    n=[z[:-2,1:-1],z[:-2,2:],z[1:-1,2:],z[2:,2:],z[2:,1:-1],z[2:,:-2],z[1:-1,:-2],z[:-2,:-2]]
    arms=sum((~n[k]&n[(k+1)%8]).astype(np.uint8) for k in range(8))
    junction=maximum_filter((sk&(arms>=3)).astype(np.uint8),size=3)>0
    junction=junction.reshape(h,sy,w,sx).any((1,3)).astype(np.float32)
    return target,valid,occupancy.astype(np.float32)[None],junction[None]


def native_centerline_target(mask,shape):
    """Downsample a native skeleton target without converting it to occupancy."""
    mask=np.asarray(mask,dtype=bool);hh,ww=mask.shape;h,w=shape
    if hh%h or ww%w:raise ValueError('Targets must align to native coarse cells')
    sy,sx=hh//h,ww//w
    sk=skeletonize(mask)
    return sk.reshape(h,sy,w,sx).any((1,3)).astype(np.float32)[None]


@torch.no_grad()
def batch_corridor_targets(target,shape):
    arrays=[native_corridor_targets(m,shape) for m in (target.detach().cpu().numpy()[:,0]>.5)]
    return tuple(torch.as_tensor(np.stack([a[j] for a in arrays]),device=target.device)
                 for j in range(4))


@torch.no_grad()
def batch_centerline_targets(target,shape):
    arrays=[native_centerline_target(m,shape) for m in (target.detach().cpu().numpy()[:,0]>.5)]
    return torch.as_tensor(np.stack(arrays),device=target.device)
