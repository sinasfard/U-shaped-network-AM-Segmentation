############# transpose Block #################

class transpose_attention(nn.Module):
  def __init__(self, in_channels, out_channels, group=1):
    super(transpose_attention,self).__init__()
    self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.cnn1 = nn.Conv2d(out_channels,out_channels, kernel_size=1, padding=0, groups=group)
    self.bn1relu1 =nn.Sequential(nn.BatchNorm2d(num_features=out_channels),
                                 nn.ReLU(inplace=True))

  def forward(self,x, y):
    #upsample
    out    = self.conv_transpose(x)
    #channel attention
    ch_att = self.xcorr_depthwise(out, y)
    out1    = out*ch_att
    out1    = self.bn1relu1(out1)
    hw = self.cnn1(out1)
    return hw

  def xcorr_depthwise(self,x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F1.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
