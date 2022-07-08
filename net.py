import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, n_class=2):
        super(Net, self).__init__()

        self.pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')
        self.pretrained_model.training = False

        # Freeze parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # self.stoplight_classifier = nn.Sequential(nn.Dropout(0.1),
        #                                           nn.Linear(1280, 80),
        #                                           nn.BatchNorm1d(80),
        #                                           nn.ReLU6(inplace=True),
        #                                           nn.Linear(80, n_class),
        #                                           nn.Softmax()
        #                                           )

        self.box_predictor = nn.Sequential(nn.Linear(1280, 80),
                                           nn.BatchNorm1d(80),
                                           nn.ReLU6(inplace=True),
                                           nn.Linear(80, 4))

    def forward(self, x):
        x = self.pretrained_model.features(x)
        x = x.mean(3).mean(2)
        return self.box_predictor(x)
