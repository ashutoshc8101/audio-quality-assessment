import torch.nn as nn
from .encoder import Encoder

class Transformer(nn.Module):

  def __init__(
    self,
    src_vocab_size,
    src_pad_index,
    embed_size = 256,
    num_layers = 6,
    forward_expansion = 4,
    heads = 8,
    dropout = 0,
    device = "cuda",
    max_length = 100
  ):
    super(Transformer, self).__init__()

    self.encoder = Encoder(
      src_vocab_size, embed_size, num_layers, heads,
      device, forward_expansion, dropout, max_length)

    self.src_pad_index = src_pad_index
    self.output = nn.Linear(src_vocab_size * embed_size, 1)
    self.device = device
    self.embed_size = embed_size
    self.src_vocab_size = src_vocab_size


  def make_src_mask(self, src):
    src_mask = (src != self.src_pad_index).unsqueeze(1).unsqueeze(2)

    # (N, 1, 1, src_len)
    return src_mask.to(self.device)


  def forward(self, src):
    src_mask = self.make_src_mask(src)
    out = self.encoder(src, src_mask)

    out = out.reshape(-1, self.src_vocab_size * self.embed_size)

    return self.output(out)