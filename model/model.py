"""MiniMind: A minimal GPT-style Transformer for LLM pre-training.

Architecture choices:
- Pre-norm with RMSNorm
- Rotary Position Embeddings (RoPE)
- SwiGLU feed-forward network
- No bias in linear layers (following LLaMA/Palm)
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    head_dim: int = 64
    vocab_size: int = 32768
    max_seq_len: int = 2048
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 500000.0) -> torch.Tensor:
    """Precompute RoPE frequencies.

    Returns (max_seq_len, dim//2) complex tensor: cos + i*sin.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)  # (max_seq_len, dim//2)
    return torch.polar(torch.ones_like(angles), angles)  # complex


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        xq: (batch, n_heads, seq_len, head_dim)
        xk: (batch, n_heads, seq_len, head_dim)
        freqs_cis: (seq_len, head_dim//2) complex tensor
    """
    # Reshape to complex: pair adjacent dims as (real, imag)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_complex = torch.view_as_complex(xq_)  # (B, H, S, head_dim//2)
    xk_complex = torch.view_as_complex(xk_)

    freqs_cis = freqs_cis[: xq.shape[2]]  # truncate to seq_len
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, S, head_dim//2)

    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dim = config.dim

        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        H, HD = self.n_heads, self.head_dim

        q = self.wq(x).view(B, S, H, HD).transpose(1, 2)  # (B, H, S, HD)
        k = self.wk(x).view(B, S, H, HD).transpose(1, 2)
        v = self.wv(x).view(B, S, H, HD).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        scale = 1.0 / math.sqrt(HD)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S, S)

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).type_as(x)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, S, HD)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(out)


class SwiGLU(nn.Module):
    """SwiGLU activation with gated linear units."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = 4 * config.dim
        # Round hidden_dim to nearest multiple of 256 for efficiency
        hidden_dim = int(2 * hidden_dim / 3)  # SwiGLU uses 2/3 hidden dim
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.attention = Attention(config)
        self.feed_forward = SwiGLU(config.dim, hidden_dim, config.dropout)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class MiniMind(nn.Module):
    """GPT-style decoder-only Transformer for pre-training."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie embedding weights
        self.token_embedding.weight = self.lm_head.weight

        # Precompute RoPE frequencies
        rope_freqs = precompute_rope_freqs(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer('rope_freqs', rope_freqs)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            attention_mask: (batch, seq_len) with 1=attend, 0=mask
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, S = input_ids.shape

        x = self.token_embedding(input_ids)

        # Build causal mask
        if attention_mask is not None:
            # Convert to boolean: True means mask (do not attend)
            causal_mask = torch.triu(
                torch.ones(S, S, device=input_ids.device, dtype=torch.bool), diagonal=1
            )
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            mask = causal_mask.unsqueeze(0) | (~padding_mask)
            mask = mask.float().masked_fill(mask, float('-inf'))
        else:
            mask = torch.triu(
                torch.full((S, S), float('-inf'), device=input_ids.device), diagonal=1
            ).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, self.rope_freqs, mask)

        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Auto-regressive generation."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            seq = input_ids[:, -self.config.max_seq_len:]

            logits = self(seq)[:, -1, :]  # (B, vocab_size)
            logits = logits / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

    def configure_optimizers(self, learning_rate: float, weight_decay: float, betas: tuple[float, float]) -> torch.optim.Optimizer:
        """AdamW with weight decay applied only to params that are not biases/norms."""
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or 'norm' in name or 'bias' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def estimate_flops_per_token(self) -> int:
        """Estimate FLOPs per token (forward pass only, no backward)."""
        n_params = self.get_num_params()
        n_non_embed = n_params - self.token_embedding.weight.numel()
        # 2 FLOPs per param per token for dense matmuls, 6 for attention + FFN
        return 6 * n_non_embed
