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

        self.kernel_time = kernel_size[1]
        self.crop = self.kernel_time - 1
        self.padding = (padding[0], self.crop)

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              groups=groups,
                              padding=self.padding,
                              bias=bias)

        # cache used for streaming inference (F, T)
        self.register_buffer("time_cache", None)
        self.initialized = False

    def forward(self, input):
        """
        Default forward method used during training and inference in non-streaming mode.
        (batch_size, in_channels, freq_bins, time_steps) --> x: (batch_size, out_channels, freq_bins, time_steps)
        """
        if getattr(self, "streaming", False) is True:
            # if streaming mode is enabled, use the streaming forward method
            return self.forward_streaming(input)
        else:
            # pytorch only includes symmetric padding --> we are removing any padding from the right hand side (future)
            x = self.conv(input)[:, :, :, :-self.crop]
            return x

    def streaming_forward(self, x_new):
        """
        Streaming forward method used during inference in streaming mode. Keeps last k-1 frames in cache, appends new input, calculates convs.
        (batch_size, in_channels, freq_bins, new_time_steps) --> x: (batch_size, out_channels, freq_bins, new_time_steps)
        """
        B, C, F, T_new = x_new.shape

        if not self.initialized:
            # init cache with zeros for warmup
            self.time_cache = torch.zeros((B, C, F, self.kernel_time-1), device=x_new.device,dtype=x_new.dtype)
            self.initialized = True

        # concatenate old cache + new input along time dim, then apply conv without time padding
        x_cat = torch.cat([self.time_cache, x_new], dim=-1)
        y_full = self.conv(x_cat)

        # remove the "future leak" frames
        y_valid = y_full[:, :, :, :-self.crop]

        # keep only the outputs that correspond to the new input frames
        out_new = y_valid[:, :, :, -(T_new):]

        # update cache (keep last k-1 time steps from x_cat)
        self.time_cache = x_cat[:, :, :, - (self.kernel_time - 1):].detach()

        return out_new


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
        self.sw_attention_window_size = int(sw_attention_window_size)
        self._block_mask_cache = {} # caching the block mask for sw_attention for efficiency

    @staticmethod
    def _force_lastdim_inner(x):
        # robustly ensure last dim is contiguous
        return x if x.stride(-1) == 1 else x.transpose(-1, -2).contiguous().transpose(-1, -2)

    @torch._dynamo.disable()  # build mask in eager to avoid ShapeAsConstantBuffer issues
    def _get_block_mask(self, q_len, kv_len, device):
        dev_key = -1 if device.type != "cuda" else (device.index or 0)
        key = (q_len, kv_len, dev_key, self.sw_attention_window_size)
        bm = self._block_mask_cache.get(key)
        if bm is not None:
            return bm

        def local_mask(b, h, i, j):
            return (j >= i - self.sw_attention_window_size) & (j <= i)

        bm = create_block_mask(
            mask_mod=local_mask,
            B=None, H=None, # for efficiency --> see FlexAttention blogpost
            Q_LEN=q_len, KV_LEN=kv_len,
            device=device,
            #_compile = True
        )
        self._block_mask_cache[key] = bm
        return bm

    def forward(self, q, k, v, use_kv_cache=False):

        if (self.sw_attention_window_size > 0) and (not use_kv_cache):
            # building block mask cache key
            b, h, q_len, _ = q.shape
            kv_len = k.shape[2]

            block_mask = self._get_block_mask(q_len, kv_len, q.device)
            q = self._force_lastdim_inner(q)
            k = self._force_lastdim_inner(k)
            v = self._force_lastdim_inner(v)

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
                q, k, v,
                dropout_p = self.dropout if self.training else 0.0,
                is_causal = False if use_kv_cache else self.causal_transformer,
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
        sw_attention_window_size=0,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.sw_attention_window_size = sw_attention_window_size

        self.attend = Attend(
            dropout=dropout,
            causal_transformer=causal_transformer,
            sw_attention_window_size=self.sw_attention_window_size
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

    @staticmethod
    def _force_lastdim_inner(x):
        # robustly ensure last dim is contiguous
        return x if x.stride(-1) == 1 else x.transpose(-1, -2).contiguous().transpose(-1, -2)

    def forward(self, x, past_kv=None, use_kv_cache=False, peek_size=None, frame_idx=0):
        x = self.norm(x)
        q, k, v = rearrange(self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads).contiguous()

        offset = frame_idx
        # apply rotary with offset-aware embedding
        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q, offset=offset)
            k = self.rotary_embed.rotate_queries_or_keys(k, offset=offset)
            #q, k, v = _check("after rope", q, k, v)

        q = self._force_lastdim_inner(q)
        k = self._force_lastdim_inner(k)
        v = self._force_lastdim_inner(v)
        #q, k, v = _check("after force-inner", q, k, v)

        # append new keys/values to past if provided
        if past_kv is not None:
            past_k, past_v = past_kv
            k_new = k[:, :, -peek_size:, :] # only the new keys/values
            v_new = v[:, :, -peek_size:, :]
            k = torch.cat([past_k, k_new], dim=2)
            v = torch.cat([past_v, v_new], dim=2)

            # prune
            if k.shape[2] > 256:
                k = k[:, :, -256:, :]
                v = v[:, :, -256:, :]

        #q, k, v = _check("before attend", q, k, v)
        out = self.attend(q, k, v, use_kv_cache=use_kv_cache)

        # optional gating
        if exists(self.to_gates):
            gates = self.to_gates(x)
            out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()

        out = rearrange(out, "b h n d -> b n (h d)")

        if use_kv_cache:
            # return new cache for caller to reuse
            return self.to_out(out), (k, v)
        else:
            return self.to_out(out)

def _check(tag, q, k, v):
    print(f"{tag}: q {q.shape} {q.stride()} | k {k.stride()} | v {v.stride()}")
    return q, k, v

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

    def forward(self, x, past_kv=None, use_kv_cache=False):
        # past_key_values: None or tuple[past_k, past_v] per layer
        new_kv = [] if use_kv_cache else None
        for i, (attn, ff) in enumerate(self.layers):
            past = past_kv[i] if (past_kv is not None and i < len(past_kv)) else None
            if use_kv_cache:
                x_attn, present = attn(x, past_kv=past, use_kv_cache=True)
                x = x + x_attn
                new_kv.append(present)
            else:
                x = x + attn(x)

            x = x + ff(x)

        x = self.norm(x)

        return (x, new_kv) if use_kv_cache else x