import torch
from torchvision import transforms

from .data.cifar import CIFAR10, CIFAR100
from .data.mnist import MNIST


def get_dataset(args):
    ### color augmentation ###
    color_jitter = transforms.ColorJitter(
        0.8 * args.color_jitter_strength,
        0.8 * args.color_jitter_strength,
        0.8 * args.color_jitter_strength,
        0.2 * args.color_jitter_strength,
    )
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225]
    )

    learning_type = args.train_type
    if learning_type == "supervised":
        learning_type = "linear_eval"

    if args.dataset == "mnist":

        if learning_type == "contrastive":
            transform_train = transforms.Compose(
                [
                    rnd_color_jitter,
                    rnd_gray,
                    transforms.RandomResizedCrop(size=28, scale=(0.4, 1)),
                    transforms.ToTensor(),
                ]
            )

            transform_test = transform_train

        elif learning_type == "linear_eval":
            transform_train = transforms.Compose(
                [
                    rnd_color_jitter,
                    rnd_gray,
                    transforms.RandomResizedCrop(size=28, scale=(0.4, 1)),
                    transforms.ToTensor(),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

        elif learning_type == "test":
            transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=28, scale=(0.4, 1)),
                    transforms.ToTensor(),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            assert "wrong learning type"

        train_dst = MNIST(
            root="./rocl/Data",
            train=True,
            download=True,
            transform=transform_train,
            contrastive_learning=learning_type,
        )
        val_dst = MNIST(
            root="./rocl/Data",
            train=False,
            download=True,
            transform=transform_test,
            contrastive_learning=learning_type,
        )

        if learning_type == "contrastive":
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dst,
                num_replicas=1,
                rank=args.local_rank,
            )
            train_loader = torch.utils.data.DataLoader(
                train_dst,
                batch_size=args.batch_size,
                num_workers=4,
                pin_memory=False,
                shuffle=True,
                sampler=train_sampler,
            )

            val_loader = torch.utils.data.DataLoader(
                val_dst,
                batch_size=100,
                num_workers=4,
                pin_memory=False,
                shuffle=True,
            )
            return train_loader, train_dst, val_loader, val_dst, train_sampler

        else:
            train_loader = torch.utils.data.DataLoader(
                train_dst, batch_size=args.batch_size, shuffle=True, num_workers=4
            )
            if "ver_total" in vars(args):
                val_loader = torch.utils.data.DataLoader(
                    val_dst, batch_size=1, shuffle=True
                )
            else:
                val_batch = 100
                val_loader = torch.utils.data.DataLoader(
                    val_dst, batch_size=val_batch, shuffle=True, num_workers=4
                )

            return train_loader, train_dst, val_loader, val_dst

    if args.dataset == "cifar-10":

        if learning_type == "contrastive":
            transform_train = transforms.Compose(
                [
                    rnd_color_jitter,
                    rnd_gray,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(32),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

            transform_test = transform_train

        elif learning_type == "linear_eval":
            transform_train = transforms.Compose(
                [
                    rnd_color_jitter,
                    rnd_gray,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(32),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        elif learning_type == "test":
            transform_train = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(32),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            assert "wrong learning type"

        train_dst = CIFAR10(
            root="./rocl/Data",
            train=True,
            download=True,
            transform=transform_train,
            contrastive_learning=learning_type,
        )
        val_dst = CIFAR10(
            root="./rocl/Data",
            train=False,
            download=True,
            transform=transform_test,
            contrastive_learning=learning_type,
        )

        if learning_type == "contrastive":
            train_loader = torch.utils.data.DataLoader(
                train_dst, batch_size=args.batch_size, shuffle=True, num_workers=4
            )
            if "ver_total" in vars(args):
                val_loader = torch.utils.data.DataLoader(
                    val_dst, batch_size=1, shuffle=True
                )
            else:
                val_loader = torch.utils.data.DataLoader(
                    val_dst, batch_size=args.batch_size, shuffle=True, num_workers=4
                )
            return train_loader, train_dst, val_loader, val_dst

    if args.dataset == "cifar-100":

        if learning_type == "contrastive":
            transform_train = transforms.Compose(
                [
                    rnd_color_jitter,
                    rnd_gray,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(32),
                    transforms.ToTensor(),
                ]
            )

            transform_test = transform_train

        elif learning_type == "linear_eval":
            transform_train = transforms.Compose(
                [
                    rnd_color_jitter,
                    rnd_gray,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(32),
                    transforms.ToTensor(),
                ]
            )

            transform_test = transforms.Compose([transforms.ToTensor()])

        elif learning_type == "test":
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )

            transform_test = transforms.Compose([transforms.ToTensor()])
        else:
            assert "wrong learning type"

        train_dst = CIFAR100(
            root="./rocl/Data",
            train=True,
            download=True,
            transform=transform_train,
            contrastive_learning=learning_type,
        )
        val_dst = CIFAR100(
            root="./rocl/Data",
            train=False,
            download=True,
            transform=transform_test,
            contrastive_learning=learning_type,
        )

        if learning_type == "contrastive":
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dst,
                num_replicas=1,
                rank=args.local_rank,
            )
            train_loader = torch.utils.data.DataLoader(
                train_dst,
                batch_size=args.batch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True,
                sampler=train_sampler,
            )

            val_loader = torch.utils.data.DataLoader(
                val_dst,
                batch_size=100,
                num_workers=4,
                pin_memory=True,
            )
            return train_loader, train_dst, val_loader, val_dst, train_sampler

        else:
            train_loader = torch.utils.data.DataLoader(
                train_dst, batch_size=args.batch_size, shuffle=True, num_workers=4
            )

            val_loader = torch.utils.data.DataLoader(
                val_dst, batch_size=100, shuffle=True, num_workers=4
            )

            return train_loader, train_dst, val_loader, val_dst



def get_train_val_test_dataset(args):
    def create_dataloader_and_split(dataset_class, transform_train, transform_test):
        dataset = dataset_class(
            root="./rocl/Data",
            train=True,
            download=True,
            transform=transform_train,
            contrastive_learning=learning_type,
        )

        # Split train dataset into training and validation sets
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        # Test set loader
        test_dataset = dataset_class(
            root="./rocl/Data",
            train=False,
            download=True,
            transform=transform_test,
            contrastive_learning=learning_type,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        return (
            train_loader,
            train_subset,
            val_loader,
            val_subset,
            test_loader,
            test_dataset,
        )
    ### Color Augmentation ###
    color_jitter = transforms.ColorJitter(
        0.8 * args.color_jitter_strength,
        0.8 * args.color_jitter_strength,
        0.8 * args.color_jitter_strength,
        0.2 * args.color_jitter_strength,
    )
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225]
    )

    learning_type = args.train_type
    if learning_type == "supervised":
        learning_type = "linear_eval"

    if args.dataset == "mnist":
        if learning_type in ["contrastive", "linear_eval"]:
            transform_train = transforms.Compose(
                [
                    rnd_color_jitter,
                    rnd_gray,
                    transforms.RandomResizedCrop(size=28, scale=(0.4, 1)),
                    transforms.ToTensor(),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

        elif learning_type == "test":
            transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=28, scale=(0.4, 1)),
                    transforms.ToTensor(),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            raise ValueError("Invalid learning type")

        return create_dataloader_and_split(MNIST, transform_train, transform_test)

    if args.dataset == "cifar-10":
        if learning_type in ["contrastive", "linear_eval"]:
            transform_train = transforms.Compose(
                [
                    rnd_color_jitter,
                    rnd_gray,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(32),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        elif learning_type == "test":
            transform_train = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(32),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            raise ValueError("Invalid learning type")

        return create_dataloader_and_split(CIFAR10, transform_train, transform_test)

    if args.dataset == "cifar-100":
        if learning_type in ["contrastive", "linear_eval"]:
            transform_train = transforms.Compose(
                [
                    rnd_color_jitter,
                    rnd_gray,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(32),
                    transforms.ToTensor(),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

        elif learning_type == "test":
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            raise ValueError("Invalid learning type")

        return create_dataloader_and_split(CIFAR100, transform_train, transform_test)


def get_cifar_10_eval_datasets(args):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if args.use_validation:
        full_train_dataset = CIFAR10(
            root="./rocl/Data",
            train=True,
            download=True,
            transform=transform_train,
            contrastive_learning="linear_eval",  # mimic default
        )

        # Split train into train + validation
        train_size = int(0.9 * len(full_train_dataset))  # 45k train / 5k val
        val_size = len(full_train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )

        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

        test_dataset = CIFAR10(
            root="./rocl/Data",
            train=False,
            download=True,
            transform=transform_test,
            contrastive_learning="linear_eval",
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

        return train_loader, train_subset, val_loader, val_subset, test_loader, test_dataset

    else:
        train_dataset = CIFAR10(
            root="./rocl/Data",
            train=True,
            download=True,
            transform=transform_train,
            contrastive_learning="linear_eval",
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )

        test_dataset = CIFAR10(
            root="./rocl/Data",
            train=False,
            download=True,
            transform=transform_test,
            contrastive_learning="linear_eval",
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

        return train_loader, train_dataset, test_loader, test_dataset