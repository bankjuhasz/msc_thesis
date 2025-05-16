"""
Transformer with rotary position embedding, adapted from Phil Wang's repository
at https://github.com/lucidrains/BS-RoFormer (under MIT License).
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import Module, ModuleList
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

# helper functions


def exists(val):
    return val is not None


# norm


class RMSNorm(Module):
    def __init__(self, size, dim=-1):
        super().__init__()
        self.scale = size**0.5
        if dim >= 0:
            raise ValueError(f"dim must be negative, got {dim}")
        self.gamma = nn.Parameter(torch.ones((size,) + (1,) * (abs(dim) - 1)))
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, dim=self.dim) * self.scale * self.gamma



# Causal Convolution


class CausalConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 bias=False):
        super().__init__()

        self.crop = kernel_size[1] - 1
        self.padding = (padding[0], self.crop)

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              groups=groups,
                              padding=self.padding,
                              bias=bias)

    def forward(self, input):
        """
        input: (batch_size, in_channels, freq_bins, time_steps)

        Returns:
        x: (batch_size, out_channels, freq_bins, time_steps)
        """
        # pytorch only includes symmetric padding --> we are removing any padding from the right hand side (future)
        x = self.conv(input)[:, :, :, :-self.crop]
        return x


# feedforward


class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.0,
        dim_out=None,
    ):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        dim_inner = int(dim * mult)
        self.activation = nn.GELU()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# attention


class Attend(nn.Module):
    def __init__(self, dropout=0.0, scale=None, causal_transformer=False, sw_attention_window_size=0):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.causal_transformer = causal_transformer
        self.sw_attention_window_size = sw_attention_window_size
        # caching the block mask for sw_attention for efficiency
        self._block_mask_cache: dict[tuple, BlockMask] = {}

    def forward(self, q, k, v):

        if self.sw_attention_window_size > 0:

            # building block mask cache key
            b, h, q_len, _ = q.shape
            cache_key = (b, h, q_len, q.device)
            # check if the block mask is already cached
            block_mask = self._block_mask_cache.get(cache_key)

            # if not cached, create the block mask
            if block_mask is None:
                # mask_mod
                def local_mask(b, h, i, j):
                    return (j >= i - self.sw_attention_window_size) & (j <= i)

                block_mask = create_block_mask(
                    mask_mod=local_mask,
                    B=None, # for efficiency --> see FlexAttention blogpost
                    H=None, # for efficiency --> see FlexAttention blogpost
                    Q_LEN=q_len,
                    KV_LEN=q_len, # key/value length --> same as query length as we're doing self-attention
                    device=q.device,
                    BLOCK_SIZE=128, #self.sw_attention_window_size,
                    _compile=True
                )
                # cache the block mask for future use
                self._block_mask_cache[cache_key] = block_mask

            # skip all-zero blocks and execute only the necessary local attention
            return flex_attention(
                q, k, v,
                block_mask=block_mask,
                scale=None
            )

        else:
            if exists(self.scale):
                default_scale = q.shape[-1] ** -0.5
                q = q * (self.scale / default_scale)

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal = self.causal_transformer # causal mask
            )


class Attention(Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        rotary_embed=None,
        gating=True,
        causal_transformer=False,
        sw_attention_window_size=0
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(
            dropout=dropout,
            causal_transformer=causal_transformer,
            sw_attention_window_size=sw_attention_window_size
        )

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        if gating:
            self.to_gates = nn.Linear(dim, heads)
        else:
            self.to_gates = None

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(
            self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        )

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        if exists(self.to_gates):
            gates = self.to_gates(x)
            out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# Roformer


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=32,
        heads=16,
        attn_dropout=0.1,
        ff_dropout=0.1,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        gating=True,
        causal_transformer=False,
        sw_attention_window_size=0
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            self.layers.append(
                ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_embed=rotary_embed,
                            gating=gating,
                            causal_transformer=causal_transformer,
                            sw_attention_window_size=sw_attention_window_size,
                        ),
                        ff,
                    ]
                )
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = self.norm(x)
        return x
