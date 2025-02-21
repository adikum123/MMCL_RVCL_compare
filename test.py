import torch
from torch.nn import ReLU, Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.padding import ZeroPad2d

from beta_crown.modules import Flatten

torch.serialization.add_safe_globals([Linear])
torch.serialization.add_safe_globals([Flatten])
torch.serialization.add_safe_globals([ReLU])
torch.serialization.add_safe_globals([Conv2d])
torch.serialization.add_safe_globals([ZeroPad2d])
torch.serialization.add_safe_globals([Sequential])
torch.serialization.add_safe_globals([set])


rvcl_model = "cnn_4layer_b"
rvcl_checkpoint = "models/unsupervised/cifar10_cnn_4layer_b.pkl"
device = torch.device("cpu")

model = torch.load(rvcl_checkpoint, device, weights_only=True)
print(model)

# Generate and test 5 random input tensors
for i in range(5):
    random_input = torch.randn(1, 3, 32, 32)  # Random tensor matching CIFAR-10 shape (batch=1, channels=3, 32x32 image)
    print(f"\nRandom Input {i+1}:")
    print(random_input)

    # Pass the tensor through the model
    output = model(random_input)
    print(f"\nModel Output {i+1}:")
    print(output)