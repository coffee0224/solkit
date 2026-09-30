"""KernelBench-style GQA attention workload for solkit.

Run:  solkit analyze examples/gqa_bench.py --arch RTX_5060_Ti --per-op
"""

import torch

B, S, Q, H, KV, D = 2, 512, 512, 16, 4, 128


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dt = torch.bfloat16
        self.wq = torch.nn.Linear(D, H * D, bias=False, dtype=dt)
        self.wk = torch.nn.Linear(D, KV * D, bias=False, dtype=dt)
        self.wv = torch.nn.Linear(D, KV * D, bias=False, dtype=dt)
        self.wo = torch.nn.Linear(H * D, D, bias=False, dtype=dt)

    def forward(self, x):
        q = self.wq(x).view(B, Q, H, D).transpose(1, 2)
        k = self.wk(x).view(B, S, KV, D).transpose(1, 2)
        v = self.wv(x).view(B, S, KV, D).transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True
        )
        return self.wo(o.transpose(1, 2).reshape(B, Q, H * D))


def get_inputs():
    return (torch.randn(B, S, D, dtype=torch.bfloat16),)
