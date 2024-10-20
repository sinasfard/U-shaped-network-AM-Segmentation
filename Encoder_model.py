import torch
from torch import nn


class Encoder_model(nn.Module):
  def __init__(self):
    super(Encoder_model, self).__init__()

    self.layer0 = s0()
    self.layer1 = s1()
    self.att = selfatt_divide4(14,56, channel=64, num_heads=14)

    self.layer2_0 = s2()
    self.multihead_attn_28_28 = selfatt_divide4(7,28, channel=128, num_heads=7)
    self.layer2 = nn.Sequential(nn.Conv2d(128,128,1),
                                nn.BatchNorm2d(num_features=128),
                                    nn.ReLU(inplace=True))

    self.layer3_0 = s3()
    self.multihead_attn_14_14 = selfatt_divide2(7, channel=256, num_heads=7)
    self.layer3 = nn.Sequential(nn.Conv2d(256,256,1),
                                nn.BatchNorm2d(num_features=256),
                                    nn.ReLU(inplace=True))

    self.layer4_0 = s4()
    self.multihead_attn_7_7 = SelfAttentionLayer(embed_dim=7*7, num_heads=7)
    self.layer4 = nn.Sequential(nn.Conv2d(512,512,1),
                                nn.BatchNorm2d(num_features=512),
                                    nn.ReLU(inplace=True))

  def forward(self, x):
    out = self.layer0(x)
    out = self.layer1(out) # torch.Size([1, 64, 56, 56])

    out = self.att(out)


    y = self.layer2_0(out) # torch.Size([1, 128, 28, 28])

    # # print(y.shape)
    y = self.multihead_attn_28_28(y) #torch.Size([1, 128, 28, 28])
    # y = y.view(batch, channel, h,w)
    y = self.layer2(y)

    y = self.layer3_0(y) #torch.Size([1, 256, 14, 14])

    y = self.multihead_attn_14_14(y)
    # y = y.view(batch, channel, h,w)
    y = self.layer3(y) #torch.Size([1, 256, 14, 14])

    y = self.layer4_0(y)
    # batch, channel, h, w = out.shape
    # y = out.view(batch, channel, h*w)#.permute(0,2,1)
    y = self.multihead_attn_7_7(y)
    y = self.layer4(y)

    return y
