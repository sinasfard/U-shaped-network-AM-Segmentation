import torch
import torch as nn

class atrous_block(nn.Module):
  def __init__(self, input_channels, output_channels, kernel_size, padding, stride, group, dilation=1):
    super(atrous_block, self).__init__()
    self.atrous1 = self._block(input_channels, output_channels, kernel_size, padding, dilation, stride, group)

  def forward(self,x):
    x = self.atrous1(x)
    return x

  def _block(self, input_channels, output_channels, kernel_size, padding, dilation, stride,group):
    module = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels= output_channels,
                                          kernel_size=kernel_size, padding= padding, dilation= dilation, stride=stride,groups=group),
                          # nn.Conv2d(output_channels,out_channels=output_channels,kernel_size=1, padding=1),
                          nn.BatchNorm2d(num_features=output_channels),
                          nn.ReLU(inplace= True)
                                 )
    return module
