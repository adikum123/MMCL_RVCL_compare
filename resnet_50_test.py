import torch
import torchvision.models as models

# Load supervised ResNet-50 pretrained on ImageNet
supervised_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
supervised_model.fc = torch.nn.Linear(in_features=2048, out_features=10, bias=True)
ssl_model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

tensor = torch.rand(1, 3, 224, 224)  # Example input tensor
supervised_output = supervised_model(tensor)
ssl_output = ssl_model(tensor)
print(supervised_output.shape)
print(ssl_output.shape)