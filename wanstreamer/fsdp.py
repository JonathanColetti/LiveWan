"""FSDP2 sharding for the block-causal student, and why it needs help.

Both trainers here are data-parallel today: Stage 1 wraps its step in a module
so DDP's hooks fire, and DMD all-reduces gradients by hand. Both keep a full
fp32 copy of the model, its gradients and its AdamW moments on every GPU --
16 bytes per parameter, 21 GB at 1.3B, and 224 GB at 14B, which is why nothing
larger than 1.3B trains on this box at all. Sharding those three tensor sets
across N ranks is the only thing that changes that, so it has to work before a
larger base is worth discussing.

THE TRAP THIS MODULE EXISTS FOR. FSDP2 all-gathers a sharded module's
parameters in a pre-forward hook installed on *that module*. `blockcausal._run`
never calls a `WanAttentionBlock` -- it passes the block to `_layer`, which
reaches into `blk.self_attn.q`, `blk.norm1`, `blk.ffn` and so on. Shard at the
block and that hook never fires, so the layer runs against parameters that are
still 1/N shards. It is structurally the same mistake DDP already invites here
("DDP does not sync gradients if you call the bare model
through helper functions"), and it is why this port is worth debugging at 1.3B.

Measured, on this codebase, the mis-shard is LOUD rather than silent: the first
parameter touched outside a forward is `blk.modulation`, and DTensor refuses
the mixed operand

    RuntimeError: aten.add.Tensor got mixed torch.Tensor and DTensor

(scripts/verify_fsdp.py check 3 asserts exactly this). That guard is DTensor's,
not ours, and it only holds while every parameter stays a DTensor; it is not a
reason to leave the shard boundary in the wrong place. The `stragglers` check
in `shard_model` is the part that does not depend on someone else's invariant.

`CausalBlock` and `CausalHead` fix it by being real modules whose forward *is*
the computation, so the shard boundary and the call boundary coincide.
`scripts/verify_fsdp.py` is the control: it asserts a sharded forward matches
the single-GPU forward, and that a deliberately mis-sharded one does not.

What is sharded: the 30 transformer blocks (98% of parameters) plus, when they
are big enough to matter, the head and the embeddings -- all of which are
entered through their own `__call__` and so need no wrapper.
"""
import torch
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, StateDictOptions
)

from . import blockcausal as bc


class CausalBlock(torch.nn.Module):
    """One WanAttentionBlock as an FSDP shard unit.

    Holds the block as a child so `fully_shard(self)` shards the block's
    parameters, and runs `_layer` in its own forward so entering the shard is
    the same act as entering the computation.
    """

    def __init__(self, blk):
        super().__init__()
        self.blk = blk

    def forward(self, x, e0, tbl, kv_ctx, ctx, ctx_lens, dtype, per_frame,
                n_frames):
        with torch.amp.autocast('cuda', enabled=False):
            ec = (self.blk.modulation + e0).chunk(6, dim=1)
        return bc._layer(self.blk, x, ec, tbl, kv_ctx, ctx, ctx_lens, dtype,
                         per_frame, n_frames)


class CausalHead(torch.nn.Module):
    """The output head as an FSDP shard unit, for the same reason: `_head`
    reaches through `model.head` to `model.head.head` and reads
    `model.head.modulation` directly."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x, e, per_frame, n_frames):
        return bc._head_from(self.head, x, e, per_frame, n_frames)


def attach_causal_modules(model):
    """Install the per-layer / head modules `blockcausal._run` will enter, and
    make each of them the *only* registered path to its parameters.

    Re-registering is the subtle half. `CausalBlock(blk)` holds `blk` as a
    child, but `model.blocks[i]` still holds it too, so every block parameter
    now has two names in the module tree. FSDP2 shards by walking that tree and
    skipping what a child FSDP module already owns; reached by its second name
    a parameter looks unmanaged, and sharding it again dies with "Cannot
    concatenate overlapping meshes". So `blocks` and `head` are demoted to
    plain attributes -- still there for `len(model.blocks)` and for the
    unsharded code path, invisible to `named_parameters()`.

    Separated from `shard_model` so the correctness control can run the wrapped
    path *without* FSDP and show that wrapping alone changes nothing.
    """
    if getattr(model, 'causal_layers', None) is not None:
        return model
    blocks = list(model.blocks)
    head = model.head
    layers = torch.nn.ModuleList(CausalBlock(b) for b in blocks)
    chead = CausalHead(head)
    del model.blocks, model.head          # drop from _modules
    model.causal_layers = layers
    model.causal_head = chead
    # object.__setattr__, because nn.Module.__setattr__ would re-register an
    # nn.Module value and undo exactly what the `del` above achieved.
    object.__setattr__(model, 'blocks', blocks)
    object.__setattr__(model, 'head', head)
    return model


def shard_model(model, mesh=None, reshard_after_forward=True, mp_policy=None,
                ignored_params=None):
    """Shard a WanModel across `mesh` with FSDP2, in place. Returns the model.

    Call BEFORE moving to device and before building the optimizer: FSDP2
    replaces `.weight` with a DTensor, and an optimizer built over the
    unsharded parameters would hold stale references.

    `reshard_after_forward=False` keeps parameters gathered between forward and
    backward. That trades memory for collectives, and it matters here more than
    in ordinary training: one DMD iteration runs the student a few dozen times
    inside its rollout, so resharding after each one pays 30 all gathers per
    forward to reclaim memory the (no_grad) rollout never needed back.

    There is deliberately no root `fully_shard(model)`. The root's hook fires
    on `model.forward`, and nothing here ever calls it. `block_forward` drives
    the model from outside. A parameter caught only by the root group would
    therefore stay sharded through every forward, silently. Everything is
    sharded at a module that is genuinely entered instead, and the assertion at
    the end refuses to return a model where that did not hold.
    """
    from torch.distributed.fsdp import fully_shard

    # An explicit, NAMED mesh. `fully_shard(mesh=None)` synthesises one per
    # call with `mesh_dim_names=None`, and composing shards over children then
    # dies in `_init_sharded_param` trying to concatenate those names.
    if mesh is None:
        from torch.distributed.device_mesh import init_device_mesh
        import torch.distributed as dist
        mesh = init_device_mesh('cuda', (dist.get_world_size(),),
                                mesh_dim_names=('dp',))
    kw = {'mesh': mesh, 'reshard_after_forward': reshard_after_forward}
    if mp_policy is not None:
        kw['mp_policy'] = mp_policy
    # `ignored_params` is how the LoRA critic survives sharding its own base.
    # The frozen teacher is bf16 with no gradients; the adapter living inside
    # the same blocks is fp32 and trainable, and it is 32 M parameters (small
    # enough that replicating it and all reducing by hand is cheaper than
    # sharding, and it keeps mixed dtypes out of a single FSDP parameter group.
    if ignored_params:
        kw['ignored_params'] = set(ignored_params)

    attach_causal_modules(model)
    units = list(model.causal_layers) + [model.causal_head]
    # These four are invoked through their own __call__ already (patch_embedding
    # in `_run`, text_embedding in the trainers, time_embedding and
    # time_projection in `time_embed`), so they need no wrapper.
    units += [model.patch_embedding, model.text_embedding,
              model.time_embedding, model.time_projection]
    for m in units:
        fully_shard(m, **kw)

    ignore = set(id(p) for p in (ignored_params or ()))
    stragglers = [n for n, p in model.named_parameters()
                  if id(p) not in ignore
                  and not isinstance(p, torch.distributed.tensor.DTensor)]
    if stragglers:
        raise RuntimeError(
            f'{len(stragglers)} parameter(s) were not sharded and are not '
            f'reachable from any entered module -- they would train '
            f'unsynchronised: {stragglers[:8]}')
    return model


def set_grad_sync(model, flag):
    """FSDP2's equivalent of `DDP.no_sync()`, over every shard unit.

    With gradient accumulation, leaving this on costs one reduce-scatter per
    micro-step instead of one per optimizer step. It is not a correctness issue
    -- reducing each micro-step and summing equals summing then reducing but
    at `--accum 4` it is four times the collectives for the same gradient.
    """
    for m in model.modules():
        if hasattr(m, 'set_requires_gradient_sync'):
            m.set_requires_gradient_sync(flag)


def full_state_dict(model):
    """Unsharded fp32 state dict on CPU, in the same key layout the existing
    checkpoints use, so `demo.py --weights` keeps working unchanged.

    Under FSDP2 `model.state_dict()` returns DTensors -- saving those would
    produce a checkpoint that only reloads onto an identical mesh, and every
    evaluation script here loads onto one GPU.
    """

    sd = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    # `attach_causal_modules` renamed the shard units, so the keys now read
    # `causal_layers.7.blk.*` and `causal_head.head.*`. Put them back under the
    # names WanModel.load_state_dict expects, so demo.py and every other
    # evaluation script keep loading these checkpoints on a single GPU.
    out = {}
    for k, v in sd.items():
        if k.startswith('causal_layers.'):
            i, rest = k[len('causal_layers.'):].split('.blk.', 1)
            k = f'blocks.{i}.{rest}'
        elif k.startswith('causal_head.head.'):
            k = 'head.' + k[len('causal_head.head.'):]
        out[k] = v
    return out
