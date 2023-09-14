import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from efficientnet_pytorch import EfficientNet

class serving_model(nn.Module):
    def __init__(self, num_classes=1196):
        super().__init__()
        self.cloud_submodel = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        self.cloud_submodel.classifier[1] = nn.Linear(1280, num_classes)

        self.control_model =  mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.control_model.classifier[3] = nn.Linear(1280, num_classes)

        self.co_submodel = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.co_submodel.classifier[3] = nn.Linear(1280, num_classes)

    def forward(self, x):
        result_co = self.co_submodel(x)
        result_cloud = self.cloud_submodel(x)
        output = result_cloud + result_co
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
    
    def load_cloudsubmodel(self, path):
        self.cloud_submodel.load_state_dict(torch.load(path))