import numpy as np

from paired_oct_registration.lateral_registration import (
    solve, estimate_bounded_lateral_d_oct, feasible_map, cell_determinants,
)
from paired_oct_registration.reconstruction import sampling_map


def parameters():
    x=np.arange(41,dtype=np.float32)
    upper=20+.1*x; lower=80+.2*x
    return dict(image_shape=np.array([101,41]),source_x_line=x+2,
                residual_target_upper=upper,residual_target_lower=lower,
                initial_target_top=upper,initial_postoperative_top=upper*1.1,
                initial_axial_scale=.9+.001*x,residual_nodes=np.zeros((64,41),np.float32),
                local_geometry_domain=np.ones(41,np.uint8))


def test_joint_path_recovers_known_axial_and_lateral_shift():
    axis=np.arange(64)
    cost=np.stack([(axis[:,None]+2-axis[None,:])**2+50*(k-3)**2 for k in range(5)],-1).astype(np.float32)
    depth,lateral=solve(cost,np.array([-4,-2,0,2,4],np.float32))
    np.testing.assert_allclose(depth[8:-8],axis[8:-8]+2)
    np.testing.assert_allclose(lateral[8:-8],2)


def test_empty_domain_retains_frozen_a_oct():
    p=parameters(); p['local_geometry_domain'][:]=0
    image=np.zeros((101,41),np.float32)
    dy,dx,stats=estimate_bounded_lateral_d_oct(image,image,p)
    assert not dy.any() and not dx.any() and not stats['d_oct_enabled']
    before=sampling_map(p)
    after=sampling_map(dict(p,residual_nodes=dy,lateral_nodes=dx,lateral_scale=1.))
    for a,b in zip(before,after): np.testing.assert_array_equal(a,b)


def test_vector_archive_composes_in_normalized_retinal_coordinates(tmp_path):
    p=parameters(); node_axis=np.sin(np.linspace(0,np.pi,64)).astype(np.float32)
    p['residual_nodes']=np.broadcast_to(node_axis[:,None],(64,41)).copy()
    p['lateral_nodes']=.5*p['residual_nodes']
    p['lateral_scale']=np.float64(.5)
    archive=tmp_path/'vector.npz'; np.savez_compressed(archive,**p)
    with np.load(archive,allow_pickle=False) as z: sx,sy=sampling_map(z)
    y,x=50,20; u=p['residual_target_upper']; h=p['residual_target_lower']-u
    node_y=u[x]+np.linspace(0,1,64)*h[x]
    dy=np.interp(y,node_y,p['residual_nodes'][:,x])
    dx=np.interp(y,node_y,p['lateral_nodes'][:,x])*.5
    qx=x+dx; cols=np.arange(41)
    qy=np.interp(qx,cols,u)+(y+dy-u[x])/h[x]*np.interp(qx,cols,h)
    expected_y=np.interp(qx,cols,p['initial_postoperative_top'])+np.interp(qx,cols,p['initial_axial_scale'])*(qy-np.interp(qx,cols,p['initial_target_top']))
    np.testing.assert_allclose([sx[y,x],sy[y,x]],[qx+2,expected_y],atol=2e-5)
    p['lateral_nodes'][:]=0
    zero=sampling_map(p); p.pop('lateral_nodes'); legacy=sampling_map(p)
    for a,b in zip(zero,legacy): np.testing.assert_array_equal(a,b)


def test_geometric_projection_reduces_folding_lateral_residual():
    p=parameters(); nodes=np.zeros((2,64,41),np.float32)
    nodes[1]=np.sin(np.linspace(0,np.pi,64))[:,None]*np.where(np.arange(41)%2,4.,-4.)[None,:]
    nodes[1,:,0]=0; nodes[1,:,-1]=0
    phi,stats=feasible_map(nodes,p)
    assert stats['raw_nonpositive_cell_fraction']>0
    assert stats['lateral_retained_scale']<1
    assert cell_determinants(phi).min()>0
