import torch
from torchvision import models

# Path to your .pth.tar file
checkpoint_path = "models/resnet_pretrained_models/resnet50_imagenet_bs2k_epochs600.pth.tar"

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract the state dictionary
state_dict = checkpoint['state_dict']

# Remove the "convnet." prefix from the keys and filter out "projection." keys
new_state_dict = {k.replace("convnet.", ""): v for k, v in state_dict.items() if not k.startswith("projection.")}

# Define the model architecture
model = models.resnet50(pretrained=False)

# Load the modified state dictionary into the model
model.load_state_dict(new_state_dict, strict=True)

# Remove the last layer (fc) by replacing it with an identity operation
model.fc = torch.nn.Identity()

# Set the model to evaluation mode
model.eval()

# Generate a random image as input
random_image = torch.randn(1, 3, 224, 224)  # Example random image

# Forward pass through the model
with torch.no_grad():
    output = model(random_image)

# Print the shape of the output tensor
print(output.shape)  # Should print the shape of the output tensor