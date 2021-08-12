import torch
import torchvision
import torch.nn as nn
from torchvision import models

class net(torch.nn.Module):
  """
    This class creates a hybrid classification-segmentation model and returns three outputs given an input tensor:
      - x: predicted segmentation masks.
      - aspp: Atrous Spatial Pyramid Pooling feature maps (encoder).
      - output: predicted classification vector.
  """
  def __init__(self):
    super().__init__()

    self.resnet = models.resnet101(pretrained=True)

    #ENCODER (RESNET101 PRETRAINED)
    self.layer1 = nn.Sequential(*list(self.resnet.children())[0:5])
    self.layer2 = nn.Sequential(*list(self.resnet.layer2))

    self.layer3 = nn.Sequential(
      nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=True),
      nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
      nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(),
      nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=True),
      nn.BatchNorm2d(512),
      nn.ReLU()
    )

    self.layer4 = nn.Sequential(
        nn.Conv2d(1024, 1024, kernel_size=1, stride=1, bias=True),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(1024),
        nn.ReLU()
    )

    #ASPP layers
    self.a1 = nn.Sequential( #1x1 Conv Stride=2
        nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=2, padding=0),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU()
    )
    self.a2 = nn.Sequential( #Dilation 2
        nn.Conv2d(in_channels= 1024, out_channels=256, kernel_size=3, stride=2, dilation=2, padding=2),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU()
    )
    self.a3 = nn.Sequential(
        nn.Conv2d(in_channels= 1024, out_channels=256, kernel_size=3, stride=2, dilation=6, padding=6),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU()
    )
    self.a4 = nn.Sequential(
        nn.Conv2d(in_channels= 1024, out_channels=256, kernel_size=3, stride=2, dilation=8, padding=8),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU()
    )
    self.a5 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels= 1024, out_channels=256, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU()
    )

    # Concatenate a_i [batch, 2048, hw]
    self.conv1 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=1280, out_channels=1024, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU()
    )

    self.upsample1 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=1536, out_channels=768, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels= 768, out_channels=768, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
    )

    self.upsample2 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels= 512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels= 512, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )

    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

    self.fc = nn.Sequential(
        nn.Linear(in_features=1280, out_features=2, bias=True),
        nn.Softmax()
    )



  def forward(self, input):

    print_shape = False #For debugging purposes

    #ENCODER--------------------------------------------------
    if(print_shape): print("Input: ", input.shape)
    l1 = self.layer1(input)
    if(print_shape): print("(l1): ", l1.shape)
    l2 = self.layer2(l1)
    if(print_shape): print("(l2): ", l2.shape)
    l3 = self.layer3(l2)
    l3 = torch.cat((l3, l2), dim=1)
    if(print_shape): print("(l3): ", l3.shape)
    l3 = self.layer4(l3)
    if(print_shape): print("(l4): ", l3.shape)

    #ATROUS SPATIAL PYRAMID POOLING
    a1 = self.a1(l3)
    if(print_shape): print("(a1) Conv1x1: ", a1.shape)
    a2 = self.a2(l3)
    if(print_shape): print("(a2) Dilation=2: ", a2.shape)
    a3 = self.a3(l3)
    if(print_shape): print("(a3) Dilation=4: ", a3.shape)
    a4 = self.a4(l3)
    if(print_shape): print("(a4) Dilation=8: ", a4.shape)
    a5 = self.a5(l3)
    if(print_shape): print("(a5) Max pool: ", a5.shape)

    aspp = torch.cat((a1, a2, a3, a4, a5), dim=1) #ASPP concatenation
    if(print_shape): print("(aspp) Concat aspp: ", aspp.shape)

    #DECODER ------------------------------------------------------------

    x = self.conv1(aspp)
    if(print_shape): print("")
    if(print_shape): print("(conv1) Conv1x1: ", x.shape)


    x = torch.cat((x, l2), dim=1) #Skip connection
    if(print_shape): print("Concat skip conv1 + l2 ", x.shape)

    x = self.upsample1(x)
    if(print_shape): print("(upsample1) up1 ", x.shape)

    x = torch.cat((x, l1), dim=1) #Skip connection
    if(print_shape): print("Concat up1 + l1: ", x.shape)

    x = self.upsample2(x)
    if(print_shape): print("(upsample2) Output: ", x.shape)

    #CAM----------------------------------------------------------------

    output = self.avgpool(aspp)
    output = output.view(output.size(0), -1)

    output = self.fc(output)

    return x, aspp, output
