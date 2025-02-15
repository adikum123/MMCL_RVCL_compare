from types import SimpleNamespace

import torch
from tqdm import tqdm

import rocl.data_loader as data_loader

model_save_name = "cnn_4layer_b_C_1_rbf_gamma.pkl"
linear_eval_save_name = f"linear_{model_save_name}"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = torch.load(f"models/mmcl/rbf/{model_save_name}", map_location=device)
linear_eval = torch.load(f"models/linear_evaluate/{linear_eval_save_name}", map_location=device)

args = {
    "dataset": "cifar-10",
    "train_type": "test",
    "color_jitter_strength": 0.5,
    "batch_size": 128
}

_, _, _, _, testloader, testdst = data_loader.get_train_val_test_dataset(SimpleNamespace(**args))

def forward(x):
    features = encoder(x)
    return linear_eval(features)

total_correct, total_samples = 0, 0
with torch.no_grad():
    for images, targets in tqdm(testloader):
        images, targets = images.to(device), targets.to(device)
        logits = forward(images)
        # Predictions
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == targets).sum().item()
        # Update metrics
        total_correct += correct
        total_samples += targets.size(0)

# print results
print(f"Test performance: {total_correct}/{total_samples} ({total_correct/total_samples*100} %)")