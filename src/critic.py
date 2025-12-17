import torch.nn as nn
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


class MobileNetV2DQN(nn.Module):
    """DQN model using MobileNetV2 backbone for feature extraction"""

    def __init__(self, num_actions=3):
        super(MobileNetV2DQN, self).__init__()

        # Load pre-trained MobileNetV2 model
        mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

        # Use all layers except the final classification layer
        self.features = mobilenet.features

        # Adaptive pooling to get fixed output size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Q-value head
        self.fc = nn.Linear(in_features=1280, out_features=num_actions)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x