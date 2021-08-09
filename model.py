import torch.nn as nn 

class unet(nn.Module):
  def __init__(self):
    super(unet, self).__init__()
    self.encoder1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1, bias=False),
                                  nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True))
    
    self.encoder2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True))
    
    self.encoder3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True))
    
    self.encoder4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True))
    
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    
    self.conv_mid = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),   
                                  nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True))
    
    self.decoder4 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True))
    
    self.decoder3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True))
    
    self.decoder2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True))
    
    self.decoder1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True))
    
    self.upconv4 = nn.ConvTranspose2d(1024, 1024, kernel_size=(2, 2), stride=2)
    
    self.upconv3 = nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
    
    self.upconv2 = nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
    
    self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
    
    self.conv1x1_out = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))

  def forward(self, x):
    out = self.encoder1(x)
    out = self.pool1(out)
    out = self.encoder2(out)
    out = self.pool2(out)
    out = self.encoder3(out)
    out = self.pool3(out)
    out = self.encoder4(out)
    out = self.pool4(out)
    out = self.conv_mid(out)
    out = self.upconv4(out)
    out = self.decoder4(out)
    out = self.upconv3(out)
    out = self.decoder3(out)
    out = self.upconv2(out)
    out = self.decoder2(out)
    out = self.upconv1(out)
    out = self.decoder1(out)
    out = self.conv1x1_out(out)

    return out