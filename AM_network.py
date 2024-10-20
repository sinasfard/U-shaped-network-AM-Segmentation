import torch
import torch as nn
import Encoder_model
from Path_Connection import atrous_block


class AM_network(nn.Module):
  def __init__(self,net, init_features=32):
    super(AM_network,self).__init__()
    #Encoder

    self.backbone = net
    self.backbone = torchvision.models._utils.IntermediateLayerGetter(self.backbone,
                                                    {
                                                    'layer0': 'out0',
                                                    'layer1': 'out1',
                                                    'layer2': 'out2',
                                                    'layer3': 'out3',
                                                    'layer4': 'out4'})

    #Bottlencek
    self.k0 = atrous_block(input_channels = 64,  output_channels=64,   kernel_size=7,   padding=2, stride=3, dilation=16, group=2)
    self.k1 = atrous_block(input_channels = 64,  output_channels=64,   kernel_size=5,   padding=2, stride=4, dilation=8,  group=4)
    self.k2 = atrous_block(input_channels = 128, output_channels=128,  kernel_size=5,   padding=1, stride=2, dilation=4,  group=4)
    self.k3 = atrous_block(input_channels = 256, output_channels=256,  kernel_size=3,   padding=2, stride=2, dilation=2,  group=4)
    self.conv_atrous = self.atrous_block(1024,512)

    #Decoder
    self.upconv4 = transpose_attention(init_features*16, init_features*8)
    self.decoder4 = self._block(init_features*16, init_features*8)

    self.upconv3 = transpose_attention(init_features*8, init_features*4)
    self.decoder3 = self._block(init_features*8, init_features*4)

    self.upconv2 = transpose_attention(init_features*4, init_features*2)
    self.decoder2 = self._block(init_features*4, init_features*2)

    self.upconv1 = nn.ConvTranspose2d(init_features*2, init_features*1, kernel_size=2, stride=2)
    self.decoder1 = self._block(init_features*2, init_features*1)

    self.upconv0 = nn.ConvTranspose2d(init_features*1, 1, kernel_size=2, stride=2)



  def forward(self, x):
    # x = input_size (batch, 3 , 224,224)
    features = self.backbone(x)

    #Encoder
    stage0 = features['out0']#(batch,64,112,112)
    # print('stage0: ',stage0.shape)

    stage1 = features['out1'] #(batch,64,56,56)
    # print('stage1: ',stage1.shape)

    stage2 = features['out2'] # batch,128,28,28)
    # print('stage2: ',stage2.shape)

    stage3 = features['out3'] # (batch,256,14,14)
    # print('stage3: ',stage3.shape)

    #Bottelneck
    stage4 = features['out4'] # (batch,512,7,7)
    # print('stage4: ',stage4.shape)

    #FPN
    atrous = torch.cat((self.k0(stage0), self.k1(stage1), self.k2(stage2), self.k3(stage3), stage4), dim=1)
    atrous = self.conv_atrous(atrous)

#Decoder
    dec4 = self.upconv4(atrous, stage3) #[1, 256, 14, 14]
    dec4 = torch.cat((dec4, stage3), dim=1) #[1, 512, 14, 14]
    dec4 = self.decoder4(dec4) #[1, 256, 14, 14]

    dec3 = self.upconv3(dec4,stage2) #[1, 128, 28, 28]
    dec3 = torch.cat((dec3, stage2), dim=1) #[1, 256, 28, 28]
    dec3 = self.decoder3(dec3) #[1, 128, 28, 28]

    dec2 = self.upconv2(dec3,stage1) #[1, 64, 56, 56]
    dec2 = torch.cat((dec2, stage1), dim=1) #[1, 128, 56, 56]
    dec2 = self.decoder2(dec2) #[1, 64, 56, 56]

    dec1 = self.upconv1(dec2) #[1, 64, 112, 112]
    dec0 = self.upconv0(dec1) #[1, 1, 224, 224]
    return dec0


  def _block(self, in_channels, features):
    module = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(num_features=features),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(num_features=features),
        nn.ReLU(inplace=True)
    )
    return module

  def atrous_block(self, in_channels, features):
    module = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(num_features=features),
        nn.ReLU(inplace=True)#,
    )
    return module
