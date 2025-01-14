import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def cifar_model():
    # cifar base
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    return model


def mnist_model_base():
    # mnist base
    model = nn.Sequential(
        nn.Conv2d(1, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(784, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    return model


def cifar_model_deep():
    # cifar deep
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    return model


def mnist_model_deep():
    # mnist deep
    model = nn.Sequential(
        nn.Conv2d(1, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(392, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    return model


def cifar_model_wide():
    # cifar wide
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    return model


def cnn_4layer():
    # cifar_cnn_a
    return cifar_model_wide()


def cnn_4layer_b():
    # cifar_cnn_b
    return nn.Sequential(
        nn.ZeroPad2d((1, 2, 1, 2)),
        nn.Conv2d(3, 32, (5, 5), stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 128, (4, 4), stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8192, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def mnist_cnn_4layer_b():
    # mnist_cnn_b
    return nn.Sequential(
        nn.ZeroPad2d((1, 2, 1, 2)),
        nn.Conv2d(1, 32, (5, 5), stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 128, (4, 4), stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(6272, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def mnist_cnn_4layer():
    # mnist_cnn_a
    return nn.Sequential(
        nn.Conv2d(1, 16, (4, 4), stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, (4, 4), stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1568, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def cut_model(model, contrastive=True, linear=False):
    if contrastive:
        return nn.Sequential(*list(model.children())[:-2])
    if linear:
        return nn.Sequential(*list(model.children())[-1])
    return model


def load_model_contrastive(args, weights_loaded=True, contrastive=True, linear=False):
    """
    Load the model architectures and weights
    """
    model_ori = eval(args.model)()
    print(f"Type of model after eval: {type(model_ori)}")
    if not weights_loaded:
        model = cut_model(model_ori, contrastive, linear)
        print(f"Type of model after cutting: {model}")
        return model
    print("loading weight...")
    map_location = None
    if args.device == "cpu":
        map_location = torch.device("cpu")

    if "cnn_4layer" not in args.model:
        model_ori.load_state_dict(
            torch.load(model_path(args), map_location)["state_dict"][0]
        )
    else:
        model_ori.load_state_dict(torch.load(model_path(args), map_location))

    return cut_model(model_ori, contrastive, linear)


def load_model_contrastive_test(
    model, model_path, device, weights_loaded=True, contrastive=True, linear=False
):
    map_location = device
    # Directly load the model since the checkpoint is not a state_dict
    try:
        return torch.load(model_path, map_location)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    return model_ori
