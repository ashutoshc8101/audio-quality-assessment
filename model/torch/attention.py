import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):

  def __init__(self, embed_size, num_heads):

      super(SelfAttention, self).__init__()

      self.embed_size = embed_size
      self.num_heads = num_heads
      self.head_dim = embed_size // num_heads

      assert (self.head_dim * num_heads ==
              embed_size), "Embed size needs to be divisible by heads"

      self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
      self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
      self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
      self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

  def forward(self, value, key, query, mask):

      N = query.shape[0]
      value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

      # Split embedding into self.num_heads pieces
      value = value.reshape(N, value_len, self.num_heads, self.head_dim)
      key = key.reshape(N, key_len, self.num_heads, self.head_dim)
      query = query.reshape(N, query_len, self.num_heads, self.head_dim)

      values = self.values(value)
      keys = self.keys(key)
      queries = self.queries(query)
      energy = torch.einsum(
          "nqhd,nkhd->nhqk", [queries, keys])  # MatMul Q and K
      # queries shape: (N, query_len, heads, heads_dim)
      # keys shape: (N, query_len, heads, heads_dim)
      # energy shape: (N, heads, query_len, key_len)

      if mask is not None:
          energy = energy.masked_fill(mask == 0, float("-1e20"))

      attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)

      out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
      out = out.reshape(
          N, query_len, self.num_heads * self.head_dim
      )

      out = self.fc_out(out)

      return out
