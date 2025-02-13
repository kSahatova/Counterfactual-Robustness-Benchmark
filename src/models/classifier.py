import numpy as np

import torch
from torch import nn

import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import Sequential, Model
import tensorflow.keras.layers as tfl 
from abc import ABC, abstractmethod


class AbstractCNN(nn.Module, ABC):
    def __init__(self, input_channels, num_classes):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.features = nn.Sequential
        self.classifier = nn.Sequential

        self.build_conv_layers()
        self.build_classifier()

    
    @abstractmethod
    def build_conv_layers(self):
        raise NotImplementedError
    
    @abstractmethod
    def build_classifier(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_params_num(self):
        raise NotImplementedError
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    

class CNNtorch(AbstractCNN):
    def __init__(self, input_channels, num_classes):
        super().__init__(input_channels, num_classes)

        self.input_channels = input_channels
        self.num_classes = num_classes

    def build_conv_layers(self):
        # input is Z, going into a convolution
        self.main = self.features(
            nn.Conv2d(self.input_channels, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.Flatten()  
        )

    def build_classifier(self):
        self.classifier = self.classifier(
            nn.Linear(32*3*3, 128),
            nn.Linear(128, self.num_classes)
        )
    
    def forward(self, x):   
            x = self.main(x)
            #x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)  # GAP Layer
            logits = self.classifier(x)
            return logits
    
    def get_params_num(self):

        features_params = self.main.parameters()
        clf_params = self.classifier.parameters()
        total_params = sum(p.numel() for p in features_params if p.requires_grad) + \
                        sum(p.numel() for p in clf_params if p.requires_grad)
        return total_params
    

class CNNtorchUpdated(nn.Module):
    def __init__(self, input_channels, img_size, num_classes):
        #super().__init__(input_channels, num_classes)
        super(CNNtorch, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.img_size = img_size
        
        self.features = nn.Sequential
        self.classifier = nn.Sequential

        self.build_conv_layers()
        self.build_classifier()
        

    def build_conv_block(self, in_ch, out_ch, kernel, stride, pad):
        return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad),
                    #nn.BatchNorm2d(out_ch),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(p=0.1)
                    )

    def build_conv_layers(self):
        # input is Z, going into a convolution
        self.in_channels = [self.input_channels, 8, 16]
        self.out_channels = [8, 16, 32]
        self.kernels = [5, 5, 3]
        conv_blocks = [self.build_conv_block(self.in_channels[i], self.out_channels[i],
                                             kernel=self.kernels[i], stride=1, pad=1)
                                               for i in range(len(self.in_channels))]
        self.main = self.features(*conv_blocks, nn.Flatten())

    def calculate_conv_output(self, width, kernel, pad=1, stride=1, is_pooling=True):
        output_shape = (width - kernel + 2*pad)/stride + 1
        if is_pooling:
             return np.floor(output_shape / 2)
        return output_shape

    def build_classifier(self):
        input_shape = self.img_size
        for kernel in self.kernels:
            output_shape = self.calculate_conv_output(input_shape, kernel)
            input_shape = int(output_shape)

        self.classifier = self.classifier(
            nn.Linear(self.out_channels[-1]*input_shape**2, 128),
            nn.Linear(128, self.num_classes)
        )
    
    def forward(self, x):   
            x = self.main(x)
            logits = self.classifier(x)
            return logits
    
    def get_params_num(self):

        features_params = self.main.parameters()
        clf_params = self.classifier.parameters()
        total_params = sum(p.numel() for p in features_params if p.requires_grad) + \
                        sum(p.numel() for p in clf_params if p.requires_grad)
        return total_params