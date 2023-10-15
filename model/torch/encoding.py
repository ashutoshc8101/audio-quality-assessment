

class PositionalEncoding(nn.Module):

  def __init__(self, embed_size, dropout, max_len = 5000):
    super().__init__()

    self.dropout = nn.Dropout(p = dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
    self.position_encoding = torch.zeros(max_len, embed_size).to(device)
    self.position_encoding[:, 0::2] = torch.sin(position * div_term).to(device)
    self.position_encoding[:, 1::2] = torch.cos(position * div_term).to(device)
    self.register_buffer('pe', self.position_encoding)

  def forward(self, x):
    x = x + self.position_encoding[:x.size(0)]
    return self.dropout(x)
