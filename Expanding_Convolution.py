import torch
import torch as nn
########################## Expanding Conv

class s0(nn.Module):
  def __init__(self):
    super(s0, self).__init__()

    self.kernel_1 = nn.Conv2d(1, 1, 1)
    self.kernel_2 = nn.Conv2d(1, 1, 1)
    self.kernel_3 = nn.Conv2d(1, 1, 1)
    self.cnn = nn.Conv2d(3, 64, 3, padding=1)
    self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    split_tensors = torch.chunk(x, 3, dim=1)
    k1 = self.kernel_1(split_tensors[0])
    k2 = self.kernel_2(split_tensors[1])
    k3 = self.kernel_3(split_tensors[2])
    # k4 = self.kernel_4(split_tensors[3])
    out = torch.cat((k1,k2,k3), dim =1)
    max = self.Maxpool(self.cnn(out))
    out = F1.relu(max)

    return out

class s1(nn.Module):
  def __init__(self):
    super(s1, self).__init__()

    self.kernel_1 = nn.Conv2d(32, 32, 1)
    self.kernel_2 = nn.Conv2d(32, 32, 1)
    self.cnn = nn.Conv2d(64, 64, 3, padding=1)
    self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    split_tensors = torch.chunk(x, 2, dim=1)
    k1 = self.kernel_1(split_tensors[0])
    k2 = self.kernel_2(split_tensors[1])
    out = torch.cat((k1,k2), dim =1)
    max = self.Maxpool(self.cnn(out))
    out = F1.relu(max)
    return out

class s2(nn.Module):
  def __init__(self):
    super(s2, self).__init__()
    self.kernel_1 = nn.Conv2d(32, 32, 1)
    self.kernel_2 = nn.Conv2d(32, 32, 1)
    self.cnn = nn.Conv2d(64, 128, 3, padding=1)
    self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    split_tensors = torch.chunk(x, 2, dim=1)
    k1 = self.kernel_1(split_tensors[0])
    k2 = self.kernel_2(split_tensors[1])
    out = torch.cat((k1,k2), dim =1)
    max = self.Maxpool(self.cnn(out))
    out = F1.relu(max)
    return out


class s3(nn.Module):
  def __init__(self):
    super(s3, self).__init__()
    self.kernel_1 = nn.Conv2d(32, 32, 1)
    self.kernel_2 = nn.Conv2d(32, 32, 1)
    self.kernel_3 = nn.Conv2d(32, 32, 1)
    self.kernel_4 = nn.Conv2d(32, 32, 1)
    self.cnn = nn.Conv2d(128, 256, 3, padding=1)
    self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    split_tensors = torch.chunk(x, 4, dim=1)
    k1 = self.kernel_1(split_tensors[0])
    k2 = self.kernel_2(split_tensors[1])
    k3 = self.kernel_3(split_tensors[2])
    k4 = self.kernel_4(split_tensors[3])
    out = torch.cat((k1,k2,k3,k4), dim =1)
    max = self.Maxpool(self.cnn(out))
    out = F1.relu(max)
    return out

class s4(nn.Module):
  def __init__(self):
    super(s4, self).__init__()
    self.kernel_1 = nn.Conv2d(128, 128, 1)
    self.kernel_2 = nn.Conv2d(128, 128, 1)
    self.cnn = nn.Conv2d(256, 512, 3, padding=1)
    self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    split_tensors = torch.chunk(x, 2, dim=1)
    # out = self.kernel_1(x)
    k1 = self.kernel_1(split_tensors[0])
    k2 = self.kernel_2(split_tensors[1])
    # k3 = self.kernel_3(split_tensors[2])
    # k4 = self.kernel_4(split_tensors[3])
    out = torch.cat((k1,k2), dim =1)

    max = self.Maxpool(self.cnn(out))
    out = F1.relu(max)

    return out
