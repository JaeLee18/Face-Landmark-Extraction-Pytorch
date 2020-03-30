## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #(W-F)/S +1
        # (32, 39, 39)
        
        # (224-5)/1 + 1 = 220 -> pool 220//2 = 110
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        
        # (110-3)/1 + 1 = 108-> pool 108//2 = 54
        self.conv2 = nn.Conv2d(32, 40, 3)
        self.conv2_bn = nn.BatchNorm2d(40)
        
        # (54-3)/1 + 1 =52 -> pool 52/2 = 26
        self.conv3 = nn.Conv2d(40, 50, 3)
        self.conv3_bn = nn.BatchNorm2d(50)
        
        # #26-3 +1 = 24  pool 24/2 = 12
        self.conv4 = nn.Conv2d(50, 128, 3)
        self.conv4_bn = nn.BatchNorm2d(128)

        # #12-3 +1 = 10  pool 10/2 = 5
        # self.conv5 = nn.Conv2d(128, 256, 3)
        # self.conv5_bn = nn.BatchNorm2d(256)

        # #5-3 +1 = 3  pool 3/2 = 1
        # self.conv6 = nn.Conv2d(256, 512, 3)
        # self.conv6_bn = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(12*12*128, 3000)
        self.fc2 = nn.Linear(3000, 136)

        self.dropout = nn.Dropout(p=0.5)
        self.conv_dp = nn.Dropout(p=0.3)
        self.pool2 = nn.MaxPool2d(2,2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.conv_dp(x)
        conv1 = self.pool2(x)

        x = self.conv2(conv1)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.conv_dp(x)
        conv2 = self.pool2(x)

        x = self.conv3(conv2)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.conv_dp(x)
        conv3 = self.pool2(x)


        x = self.conv4(conv3)
        x = self.conv4_bn(x)
        x = F.relu(x)
        x = self.conv_dp(x)
        conv4 = self.pool2(x)

        # x = self.conv5(conv4)
        # x = self.conv5_bn(x)
        # x = F.relu(x)
        # x = self.dropout(x)
        # conv5 = self.pool2(x)

        # x = self.conv6(conv5)
        # x = self.conv6_bn(x)
        # x = F.relu(x)
        # x = self.dropout(x)
        # conv6 = self.pool2(x)

        
        x = conv4.view(conv4.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        fc1 = self.dropout(x)

        x = self.fc2(fc1)



        
        # a modified x, having gone through all the layers of your model, should be returned
        return x, conv1, conv2, conv3, conv4
