import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, input_size=100, output_size=784) -> None:
        super().__init__() 
        self.input_size = input_size
        self.output_size = output_size

        self.layer1 = nn.Linear(in_features=self.input_size, out_features=128)
        self.layer2 = nn.Linear(in_features=128, out_features=256)
        self.layer3 = nn.Linear(in_features=256, out_features=512)
        self.layer4 = nn.Linear(in_features=512, out_features=1024)
        self.output_layer = nn.Linear(in_features=1024, out_features=self.output_size)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x), 0.2)
        x = F.leaky_relu(self.layer2(x), 0.2)
        x = F.leaky_relu(self.layer3(x), 0.2)
        x = F.leaky_relu(self.layer4(x), 0.2)
        x = torch.tanh(self.output_layer(x))  
        return x



class Discriminator(nn.Module):
    def __init__(self, input_size=784, output_size=1) -> None:
        super().__init__() 
        self.input_size = input_size
        self.output_size = output_size

        # self.layer1 = nn.Linear(in_features=self.input_size, out_features=1024)
        # self.layer2 = nn.Linear(in_features=1024, out_features=512)
        # self.layer3 = nn.Linear(in_features=512, out_features=256)
        # self.output_layer = nn.Linear(in_features=256, out_features=self.output_size)

        self.layer1 = nn.Sequential(
                    nn.Linear(self.input_size, 1024),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
        self.layer2 = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
        self.layer3 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
        self.output_layer = nn.Sequential(
                    nn.Linear(256, self.output_size),
                    nn.Sigmoid()
                    )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x
