import torch
from torch import nn
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms import transforms


class MobileNet(nn.Module):
    def __init__(self, n_class=2):
        super(MobileNet, self).__init__()

        self.pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.pretrained_model.training = False

        # Freeze parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.stoplight_classifier = nn.Sequential(nn.Linear(1280, 160),
                                                  nn.BatchNorm1d(160),
                                                  nn.ReLU6(inplace=True),
                                                  nn.Linear(160, n_class),
                                                  nn.Softmax()
                                                  )

        self.box_predictor = nn.Sequential(nn.Linear(1280, 80),
                                           nn.BatchNorm1d(80),
                                           nn.ReLU6(inplace=True),
                                           nn.Linear(80, 4))

    def forward(self, x):
        x = self.pretrained_model.features(x)
        x = x.mean(3).mean(2)
        return self.stoplight_classifier(x), self.box_predictor(x)
