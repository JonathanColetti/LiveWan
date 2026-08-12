"""CUDA-graph capture for the streaming forward.

Why this and not a smaller optimisation: the prior profiling of this pipeline
found ~7,300 kernel launches and only ~17% arithmetic in a single frame-forward,
and the sweep here reproduces the signature exactly -- a 1-latent-frame unit and
a 3-latent-frame unit cost nearly the same per forward (119 ms vs 168 ms) even
though the second does 3x the work. The bottleneck is launch overhead, so the
fix has to remove launches, not work. A CUDA graph replays the whole forward as
one submission.

The capture is only valid because streaming reaches a genuine steady state:

  * the event K/V window is bounded, so once it is full `cache.length` stops
    changing and every shape in the forward is constant;
  * the cache is one preallocated buffer written in place at a constant offset,
    so the graph can own those writes;
  * only three things vary between calls -- the latents, the timestep, and the
    RoPE table slice for the current absolute frame index -- and all three are
    copied into static input buffers before replay.

So: run eagerly until the window fills, capture once, then replay. Anything that
would change a shape (a different block size, the window not yet full) falls
back to eager automatically.
"""
import torch


class GraphedForward:
    """Captures `fn(z, tbl, e, e0) -> v` for one fixed streaming configuration."""

    def __init__(self, fn, z_shape, tbl_shape, e_shape, e0_shape, device,
                 dtype=torch.float32, warmup=3):
        self.fn = fn
        self.device = device
        self.z = torch.zeros(z_shape, device=device, dtype=dtype)
        self.tbl = torch.zeros(tbl_shape, device=device, dtype=torch.complex64)
        self.e = torch.zeros(e_shape, device=device, dtype=torch.float32)
        self.e0 = torch.zeros(e0_shape, device=device, dtype=torch.float32)
        self.graph = None
        self.warmup = warmup
        self.out = None
        self.key = None

    def capture(self):
        # Warm up on a side stream first: cuDNN/cuBLAS pick algorithms and
        # allocate workspaces on first call, and that must not happen during
        # capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                out = self.fn(self.z, self.tbl, self.e, self.e0)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self.fn(self.z, self.tbl, self.e, self.e0)
        return self

    def __call__(self, z, tbl, e, e0):
        self.z.copy_(z)
        self.tbl.copy_(tbl)
        self.e.copy_(e)
        self.e0.copy_(e0)
        self.graph.replay()
        return self.out


class SteadyStateGraphs:
    """One graph per distinct (cache length, block size) the stream settles into.

    A stream with a bounded window visits at most a handful of cache lengths
    before it saturates, so keying on the length is enough; the dict never grows
    without bound. `enabled=False` makes every call fall through to eager, which
    is what the correctness comparison in scripts/bench_graph.py uses.
    """

    def __init__(self, enabled=True, max_graphs=4):
        self.enabled = enabled
        self.graphs = {}
        self.max_graphs = max_graphs
        self.captures = 0
        self.replays = 0

    def get(self, key, make):
        if not self.enabled:
            return None
        g = self.graphs.get(key)
        if g is None:
            if len(self.graphs) >= self.max_graphs:
                return None
            g = make().capture()
            self.graphs[key] = g
            self.captures += 1
        self.replays += 1
        return g
