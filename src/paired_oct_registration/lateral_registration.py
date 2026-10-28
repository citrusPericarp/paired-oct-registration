"""Gray/gradient DTW with bounded lateral residuals and spatial smoothing."""
from __future__ import annotations
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d
from . import registration_core as core
from .registration_utils import normalize_percentile, BoundaryData
from .reconstruction import sampling_map

OFFSETS = np.array([-4,-2,0,2,4], dtype=np.float32)

@njit(cache=True)
def solve(cost, offsets):
    """Float32 accumulated DP; same tie order as original for one x state."""
    n,m,kmax = cost.shape
    dp = np.full((n,m,kmax),np.float32(1e30),np.float32)
    backstep = np.full((n,m,kmax),-1,np.int8)
    backstate = np.full((n,m,kmax),-1,np.int8)
    center = kmax//2
    dp[0,0,center] = cost[0,0,center]
    for i in range(n):
        for j in range(max(0,i-12),min(m,i+13)):
            if i==0 and j==0:
                continue
            for k in range(kmax):
                best = 1e30
                bs,bk = -1,-1
                for step in range(3):
                    pi = i-(step!=2)
                    pj = j-(step!=1)
                    if pi<0 or pj<0:
                        continue
                    # Prefer retaining x for a tied path.
                    for order in range(3):
                        pk = k if order==0 else (k-1 if order==1 else k+1)
                        if pk<0 or pk>=kmax:
                            continue
                        penalty = (0.0 if step==0 else 0.10) + (0.0 if pk==k else 0.10)
                        value = float(dp[pi,pj,pk])+penalty
                        if value<best:
                            best,bs,bk = value,step,pk
                if bs>=0:
                    prior = 0.01*(float(offsets[k])/4.0)**2
                    dp[i,j,k] = float(cost[i,j,k])+best+prior
                    backstep[i,j,k],backstate[i,j,k] = bs,bk
    # Match production anchors at both boundaries.
    i,j,k = n-1,m-1,center
    sums = np.zeros(n,np.float64); xs = np.zeros(n,np.float64); counts = np.zeros(n,np.int32)
    while i>=0 and j>=0:
        sums[i]+=j; xs[i]+=offsets[k]; counts[i]+=1
        if i==0 and j==0:
            break
        step,pk = backstep[i,j,k],backstate[i,j,k]
        if step<0:
            break
        i -= step!=2; j -= step!=1; k=pk
    mapping = np.arange(n,dtype=np.float32)
    dx = np.zeros(n,np.float32)
    for i in range(n):
        if counts[i]>0:
            mapping[i]=sums[i]/counts[i]; dx[i]=xs[i]/counts[i]
    return mapping,dx

def regularize(raw,p,smooth):
    result = raw.copy(); w=raw.shape[-1]; xaxis=np.arange(w,dtype=np.float32)
    for component in range(2):
        for node in range(64):
            row=result[component,node]; valid=np.isfinite(row)
            known=xaxis[valid].copy(); values=row[valid].copy(); row[:]=0
            if valid.sum()>1:
                lo,hi=int(known[0]),int(known[-1]); row[lo:hi+1]=np.interp(xaxis[lo:hi+1],known,values).astype(np.float32)
    if smooth:
        result=gaussian_filter1d(result,sigma=2,axis=1,mode='nearest')
        result=gaussian_filter1d(result,sigma=6,axis=2,mode='nearest')
    _,weight,_,_=core.safe_geometry_domain(p['local_geometry_domain'].astype(bool))
    for component,limit in enumerate([12,4]):
        result[component]=np.clip(result[component],-limit,limit)*weight[None,:]
    result[:,[0,-1],:]=0
    boundary=BoundaryData(p['residual_target_upper'],p['residual_target_lower'],p['local_geometry_domain'].astype(bool),{})
    result[0]=core.enforce_monotone_residual_nodes(result[0],boundary)
    return result

def cell_determinants(phi):
    p=phi.astype(np.float64)
    b=p[:-1,1:]-p[:-1,:-1]; c=p[1:,:-1]-p[:-1,:-1]
    d=p[1:,1:]-p[:-1,1:]-p[1:,:-1]+p[:-1,:-1]
    cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
    return np.minimum.reduce([cross(b,c),cross(b,c+d),cross(b+d,c),cross(b+d,c+d)])

def feasible_map(nodes,p):
    scale=1.; phi=compose(nodes,p,scale); det=cell_determinants(phi)
    raw_fraction=float((det<=0).mean()); raw_min=float(det.min())
    if np.any(nodes[1]):
        while det.min()<=1e-6 and scale>1/1024:
            scale/=2; phi=compose(nodes,p,scale); det=cell_determinants(phi)
        if det.min()<=1e-6:
            scale=0.; phi=compose(nodes,p,scale); det=cell_determinants(phi)
    assert np.isfinite(phi).all() and det.min()>0,(scale,float(det.min()))
    return phi,dict(lateral_retained_scale=scale,raw_nonpositive_cell_fraction=raw_fraction,
                    raw_minimum_cell_jacobian=raw_min,minimum_cell_jacobian=float(det.min()),
                    final_nonpositive_cell_fraction=float((det<=0).mean()))

def estimate_nodes(a, moving, p):
    height, width = a.shape
    upper, lower = p['residual_target_upper'], p['residual_target_lower']
    domain, _, _, _ = core.safe_geometry_domain(p['local_geometry_domain'].astype(bool))
    fixed, shifted = normalize_percentile(a), normalize_percentile(moving)
    axis = np.arange(64, dtype=np.float32)
    depth_axis = np.arange(height, dtype=np.float32)
    nodes = np.full((2,64,width), np.nan, dtype=np.float32)
    for x in np.flatnonzero(domain):
        thickness = float(lower[x])-float(upper[x])
        if not 24 <= thickness <= 240:
            continue
        ys = np.linspace(float(upper[x]),float(lower[x]),64,dtype=np.float32)
        ref = core.zscore(np.interp(ys,depth_axis,fixed[:,x]))
        grad = core.zscore(np.gradient(ref))
        cost = np.full((64,64,len(OFFSETS)),np.float32(1e20),np.float32)
        for state, dx in enumerate(OFFSETS):
            mx = x+int(dx)
            if not 0 <= mx < width or not domain[mx]:
                continue
            moving_y = ys if dx == 0 else np.linspace(float(upper[mx]),float(lower[mx]),64,dtype=np.float32)
            values = core.zscore(np.interp(moving_y,depth_axis,shifted[:,mx]))
            moving_grad = core.zscore(np.gradient(values))
            cost[:,:,state] = .72*(ref[:,None]-values[None,:])**2 + .28*(grad[:,None]-moving_grad[None,:])**2
        mapping, dx = solve(cost,OFFSETS)
        dy = (mapping-axis)*(thickness/63)
        dy -= np.linspace(dy[0],dy[-1],64,dtype=np.float32)
        dy[[0,-1]]=0; dx[[0,-1]]=0
        nodes[0,:,x] = np.clip(dy,-12,12)
        nodes[1,:,x] = np.clip(dx,-4,4)
    return nodes


def compose(nodes,p,lateral_scale=1.):
    return np.stack(sampling_map(dict(p,residual_nodes=nodes[0],lateral_nodes=nodes[1],
                                      lateral_scale=np.float64(lateral_scale))),-1)


def estimate_bounded_lateral_d_oct(a, a_oct, parameters):
    """Return axial nodes, lateral nodes and geometry diagnostics; no labels."""
    raw = estimate_nodes(a,a_oct,parameters)
    nodes = regularize(raw,parameters,True)
    _, stats = feasible_map(nodes,parameters)
    stats.update(d_oct_enabled=float(np.asarray(parameters['local_geometry_domain']).any()),
                 d_oct_valid_columns=int(np.asarray(parameters['local_geometry_domain']).sum()))
    return nodes[0], nodes[1], stats
