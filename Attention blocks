############################## Attention block ###################
class SelfAttentionLayer(nn.Module):
  def __init__(self, embed_dim, num_heads):
      super(SelfAttentionLayer, self).__init__()
      self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

  def forward(self, x):
      # Ensure input is reshaped properly for multi-head attention
      batch_size, channels, height, width = x.size()
      x = x.reshape(batch_size, channels, height * width)  # Reshape to batch_size x channels x (height * width)
      x = self.multihead_attn(x, x, x)[0]
      x = x.view(batch_size, channels, height,width)
      # Assuming x is both the query, key, and value
      return x

class selfatt_divide4(nn.Module):
  def __init__(self,x1,x2, channel, num_heads):
    super(selfatt_divide4,self).__init__()
    self.att1 = SelfAttentionLayer(embed_dim=x1*x1, num_heads=num_heads)
    self.att2 = SelfAttentionLayer(embed_dim=x1*x1, num_heads=num_heads)
    self.att3 = SelfAttentionLayer(embed_dim=x1*x1, num_heads=num_heads)
    self.att4 = SelfAttentionLayer(embed_dim=x1*x1, num_heads=num_heads)
    self.conv = nn.Conv2d(channel,channel,1)
    self.x1 = x1
    self.x2 = x2

  def forward(self,x):
    o1 = x[:, :, :self.x1, :self.x1]
    o2 = x[:, :, :self.x1, self.x2-self.x1:]
    o3 = x[:, :, self.x2-self.x1:, :self.x1]
    o4 = x[:, :, self.x2-self.x1:, self.x2-self.x1:]

    o1 = self.att1(o1)
    o2 = self.att2(o2)
    o3 = self.att3(o3)
    o4 = self.att4(o4)

    combined_tensor1 = torch.cat((o1, o2, o3, o4), dim=2)
    combined_tensor2 = torch.cat((o1, o2, o3, o4), dim=3)

    combined_tensor3 = torch.cat((combined_tensor1, combined_tensor1, combined_tensor1, combined_tensor1), dim=3)
    combined_tensor4 = torch.cat((combined_tensor2, combined_tensor2, combined_tensor2, combined_tensor2), dim=2)

    out = combined_tensor3 * combined_tensor4
    out = self.conv(out)
    return out

class selfatt_divide2(nn.Module):
  def __init__(self,x1, channel, num_heads):
    super(selfatt_divide2,self).__init__()
    self.att1 = SelfAttentionLayer(embed_dim=x1*x1, num_heads=num_heads)
    self.att2 = SelfAttentionLayer(embed_dim=x1*x1, num_heads=num_heads)
    self.conv = nn.Conv2d(channel,channel,1)
    self.x1 = x1

  def forward(self,x):
    o1 = x[:,:,self.x1 :, self.x1 :]
    o2 = x[:,:,:self.x1 , :self.x1 ]

    o1 = self.att1(o1)
    o2 = self.att2(o2)

    resized_Tensor_1 = torch.cat((o1, o2), dim=2)
    resized_Tensor_2 = torch.cat((o1, o2), dim=3)

    resized_Tensor_1 = torch.cat((resized_Tensor_1, resized_Tensor_1), dim=3)
    resized_Tensor_2 = torch.cat((resized_Tensor_2, resized_Tensor_2), dim=2)

    out = resized_Tensor_1 * resized_Tensor_2
    out = self.conv(out)
    return out
