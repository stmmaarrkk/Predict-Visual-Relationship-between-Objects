import torch
import torch.nn as nn
from ipdb import set_trace

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        #self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        #output = self.bn(output)
        output = self.relu(output)

        return output


class color_clf_cnn(nn.Module):
    def __init__(self, num_classes=6):
        first_sec = 64
        second_sec = 2 * first_sec #128
        third_sec = 2 * second_sec #256
        forth_sec = 2 * third_sec #512
        super(color_clf_cnn, self).__init__()
        # Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3, out_channels=first_sec)
        self.unit2 = Unit(in_channels=first_sec, out_channels=first_sec)
        #self.unit3 = Unit(in_channels=first_sec, out_channels=first_sec)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=first_sec, out_channels=second_sec)
        self.unit5 = Unit(in_channels=second_sec, out_channels=second_sec)
        self.unit6 = Unit(in_channels=second_sec, out_channels=second_sec)
        #self.unit7 = Unit(in_channels=second_sec, out_channels=second_sec)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=second_sec, out_channels=third_sec)
        self.unit9 = Unit(in_channels=third_sec, out_channels=third_sec)
        self.unit10 = Unit(in_channels=third_sec, out_channels=third_sec)
        #self.unit11 = Unit(in_channels=third_sec, out_channels=third_sec)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.unit12 = Unit(in_channels=third_sec, out_channels=forth_sec)
        self.unit13 = Unit(in_channels=forth_sec, out_channels=forth_sec)
        #self.unit14 = Unit(in_channels=forth_sec, out_channels=forth_sec)

        #self.avgpool = nn.AvgPool2d(kernel_size=4)

        self.linear = nn.Linear(in_features=forth_sec, out_features=4096)


        # Add all the units into the Sequential layer in exact order
        #self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1,self.fc)
        """
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6,
                                 self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)
        """
        self.net = nn.Sequential(self.unit1, self.unit2, self.pool1, self.unit4, self.unit5, self.unit6,
                                 self.pool2, self.unit8, self.unit9, self.unit10, self.pool3,
                                 self.unit12, self.unit13)

    def forward(self, input):
        #print("pool {0}".format(input.shape))
        ###another
        output = input.view(-1, 3, 7, 7)
        output = self.net(output)
        output = output.view(-1, 512) #the second dim must be same as the final_sec of network
        output = self.linear(output)
        """
        output = self.unit1(output)
        output = self.unit2(output)
        output = self.unit3(output)
        output = self.pool1(output)
        output = output.view(-1, 288)
        output = self.fc(output)
        #print("pass all")
        #print("output: {0}".format(output.shape))
        """
        return output
