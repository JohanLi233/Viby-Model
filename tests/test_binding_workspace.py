import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from model.binding_workspace import BindingConfig, BindingWorkspace, ball, memory_scan
from test_dpr import cfg
from _v41_common import build


def close(a, b, atol=2e-5):
    np.testing.assert_allclose(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)), atol=atol, rtol=2e-5)


@pytest.mark.parametrize("length", [1, 3, 8, 17])
def test_scan_values_gradients_reset_padding_and_initial_state(length):
    mx.random.seed(88)
    k, v = ball(mx.random.normal((2, length, 8))), ball(mx.random.normal((2, length, 8)))
    beta = mx.sigmoid(mx.random.normal((2, length, 2)))
    valid = mx.arange(length)[None] != 2
    valid = mx.broadcast_to(valid, (2, length))
    reset = mx.broadcast_to((mx.arange(length) == 4)[None], (2, length))
    initial = mx.random.normal((2, 2, 8, 8)) * .1
    def loss(k, v, beta, init, ref):
        return mx.sum(memory_scan(k, v, beta, valid, reset, init, reference=ref)**2)
    close(memory_scan(k, v, beta, valid, reset, initial), memory_scan(k, v, beta, valid, reset, initial, reference=True))
    a = mx.grad(lambda k,v,b,i: loss(k,v,b,i,False), argnums=(0,1,2,3))(k,v,beta,initial)
    b = mx.grad(lambda k,v,b,i: loss(k,v,b,i,True), argnums=(0,1,2,3))(k,v,beta,initial)
    for x, y in zip(a,b):
        close(x,y)
    if length > 4:
        changed = memory_scan(mx.concatenate([-k[:, :4],k[:, 4:]],1), v, beta, valid, reset, initial)
        close(changed[:, 4:], memory_scan(k,v,beta,valid,reset,initial)[:,4:])


def test_exact_bindings_overwrite_and_composition():
    basis = mx.eye(4)
    # f(A)=B; g(B)=C; overwrite f(A)=D, with bank-specific write gates.
    k = mx.stack([basis[0],basis[1],basis[0]])[None]
    v = mx.stack([basis[1],basis[2],basis[3]])[None]
    rates = mx.array([[[1.,0.],[0.,1.],[1.,0.]]])
    states = memory_scan(k,v,rates,mx.ones((1,3),mx.bool_),mx.zeros((1,3),mx.bool_))
    q1=states[0,1,0]@basis[0]
    close(q1,basis[1]); close(states[0,1,1]@q1,basis[2])
    close(states[0,1,1]@basis[0],mx.zeros((4,)))
    close(states[0,2,0]@basis[0],basis[3])
    close(ball(mx.zeros((4,))),mx.zeros((4,)))
    assert bool(mx.all(mx.isfinite(mx.grad(lambda x:ball(x).sum())(mx.zeros((4,))))))


@pytest.mark.parametrize("compiled", [False, True])
def test_causal_gradient_wiring_and_fixed_bridge(compiled):
    model=build(cfg(dpr_enabled=False))
    model.binding=BindingWorkspace(BindingConfig(dim=64,rank=8))
    model.train()
    x=mx.array([[1,8,9,10,11,12,13,14]])
    params=model.trainable_parameters()
    assert 'bridge' not in params['binding']
    def fn(p):
        model.update(p)
        return model(x,labels=x,use_dpr=False,use_ced_recurrent=False).loss
    grad=mx.value_and_grad(fn)
    if compiled: grad=mx.compile(grad)
    loss,g=grad(params);mx.eval(loss,g);model.update(params)
    flat=dict(tree_flatten(g))
    assert bool(mx.isfinite(loss))
    assert all(bool(mx.all(mx.isfinite(v))) for v in flat.values())
    assert float(mx.linalg.norm(flat['binding.key.weight']))>0
    assert float(mx.linalg.norm(flat['model.embed.weight']))>0
    model.eval()
    y=model(x,use_dpr=False,use_ced_recurrent=False).logits
    changed=model(x.at[:,4:].add(20),use_dpr=False,use_ced_recurrent=False).logits
    close(y[:,:4],changed[:,:4],atol=3e-4)
    p=model.binding.bridge
    close(p.T@p,mx.eye(16))
    injection,_,_=model.binding(mx.random.normal((1,8,64)),mx.random.normal((1,8,64)),x)
    assert float(mx.max(mx.linalg.norm(injection,axis=-1))) <= 2**.5+1e-5


@pytest.mark.parametrize("mode", ["full","fixed"])
def test_native_cached_prefill_decode_and_condition_boundary(mode):
    model=build(cfg(dpr_enabled=False))
    model.binding=BindingWorkspace(BindingConfig(dim=64,rank=8),mode)
    model.eval()
    x=mx.array([[1,8,9,10,11,12,13,14]])
    full=model(x,use_dpr=False,use_ced_recurrent=False).logits
    pre,cache=model.prefill(x[:,:3],use_dpr=False,use_ced_recurrent=False)
    pieces=[pre]
    for t in range(3,8):
        step,cache=model.decode_step(x[:,t],cache);pieces.append(step[:,None])
    close(full,mx.concatenate(pieces,1),atol=5e-4)
    with pytest.raises(ValueError,match="rewind"):
        cache.rewind(1)
    model.binding.address_mode="fixed" if mode=="full" else "full"
    with pytest.raises(ValueError,match="fresh cache"):
        model.decode_step(mx.array([15]),cache)


def test_address_control_matched_parameters_and_effect():
    c=BindingConfig(dim=64,rank=8)
    a,b=BindingWorkspace(c,"full"),BindingWorkspace(c,"fixed")
    for (ka,va),(kb,vb) in zip(tree_flatten(a.parameters()),tree_flatten(b.parameters())):
        assert ka==kb;close(va,vb,atol=0)
    e=mx.random.normal((1,12,64));q=mx.random.normal(e.shape);x=mx.ones((1,12),mx.int32)
    full=a(e,q,x);fixed=b(e,q,x)
    close(full[1],fixed[1]);close(full[2][0],fixed[2][0])
    assert float(mx.max(mx.abs(full[2][1]-fixed[2][1])))>1e-4
    big=BindingWorkspace()
    assert sum(v.size for _,v in tree_flatten(big.trainable_parameters()))==202886


def test_bf16_backbone_fp32_binding_gradient_and_roundtrip(tmp_path):
    from trainer.utils import convert_model_dtype
    model=convert_model_dtype(build(cfg(dpr_enabled=False)),"bfloat16")
    model.binding=BindingWorkspace(BindingConfig(dim=64,rank=8))
    x=mx.array([[1,8,9,10,11,12,13,14]])
    params=model.trainable_parameters()
    def f(p):
        model.update(p)
        return model(x,labels=x,use_dpr=False,use_ced_recurrent=False).loss
    loss,g=mx.value_and_grad(f)(params);mx.eval(loss,g);model.update(params)
    assert bool(mx.isfinite(loss))
    assert g['binding']['key']['weight'].dtype==mx.float32
    assert float(mx.linalg.norm(g['binding']['key']['weight']))>0
    path=str(tmp_path/'workspace.safetensors')
    model.binding.save_weights(path)
    other=BindingWorkspace(model.binding.config)
    other.load_weights(path,strict=True)
    for (_,a),(_,b) in zip(tree_flatten(other.parameters()),tree_flatten(model.binding.parameters())):
        close(a,b,atol=0)
