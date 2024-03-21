import torch
import torch.nn as nn

from model import ViT
from train import train
from utils.utils import *
from utils.optimizer import get_adam_optimizer
from utils.parser import parser
from ema import ExponentialMovingAverage


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Rotation ViT
def main():
    args, unknown = parser.parse_known_args()
    args.checkpoint_dir = '/content/assignment3/assignment3/assignment3_part1/Checkpoints'
    args.seed = 315

    setup_seed(args.seed)
    print_gpu_info(device)

    print("\n--- Preparing Data for Rotation Task ---\n")

    transform_train, transform_test = get_transforms()

    trainset = CIFAR10Rotation(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.BATCH_SIZE,
                                              shuffle=True, num_workers=args.NUM_WORKERS)

    testset = CIFAR10Rotation(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.BATCH_SIZE,
                                            shuffle=False, num_workers=args.NUM_WORKERS)

    print("\n--- Training ViT on Rotation Task ---\n")

    model_path = '/content/assignment3/assignment3/assignment3_part1/ViT_rotation_task_best_model.pth'
    project_name = 'cs_mp3'

    model = ViT(args.NUM_CLASSES, args.IMAGE_SIZE, channels=args.CHANNELS, head_channels=args.HEAD_CHANNELS, num_blocks=args.NUM_BLOCKS, patch_size=args.PATCH_SIZE,
               emb_p_drop=args.EMB_P_DROP, trans_p_drop=args.TRANS_P_DROP, head_p_drop=args.HEAD_P_DROP).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_adam_optimizer(model.parameters(), lr=args.LEARNING_RATE, wd=args.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.LEARNING_RATE, steps_per_epoch=len(trainloader), epochs=args.EPOCHS)

    train(model, criterion, optimizer, ema=None, num_epochs=args.EPOCHS, lr_scheduler=lr_scheduler, init_lr=args.LEARNING_RATE, task='rotation', trainloader=trainloader, testloader=testloader, model_path=model_path)

if __name__ == "__main__":
    main()


# Classification ViT
def main():
    args, unknown = parser.parse_known_args()
    args.checkpoint_dir = '/content/assignment3/assignment3/assignment3_part1/Checkpoints'
    args.seed = 315

    setup_seed(args.seed)
    print_gpu_info(device)

    print("\n--- Preparing Data for Classification Task ---\n")

    transform_train, transform_test = get_transforms()

    trainset = CIFAR10Rotation(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.BATCH_SIZE,
                                              shuffle=True, num_workers=args.NUM_WORKERS)

    testset = CIFAR10Rotation(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.BATCH_SIZE,
                                            shuffle=False, num_workers=args.NUM_WORKERS)

    print("\n--- Training ViT on Classification Task ---\n")

    path = '/content/assignment3/assignment3/assignment3_part1/ViT_rotation_task_best_model.pth'
    model_path = '/content/assignment3/assignment3/assignment3_part1/ViT_classification_task_best_model.pth'
    project_name = 'cs_mp3'

    def replace_last_linear_layer(model, num_classes):
        for name, module in reversed(model._modules.items()):
            if hasattr(module, 'children') and list(module.children()):
                replace_last_linear_layer(module, num_classes)
                return
            if isinstance(module, nn.Linear):
                num_ftrs = module.in_features
                setattr(model, name, nn.Linear(num_ftrs, num_classes))
                print(f"Replaced Linear Layer: {name}")
                return

    model = ViT(args.NUM_CLASSES, args.IMAGE_SIZE, channels=args.CHANNELS, head_channels=args.HEAD_CHANNELS, num_blocks=args.NUM_BLOCKS, patch_size=args.PATCH_SIZE,
               emb_p_drop=args.EMB_P_DROP, trans_p_drop=args.TRANS_P_DROP, head_p_drop=args.HEAD_P_DROP).to(device)

    model.load_state_dict(torch.load(path), strict=False)

    new_num_classes = 10
    replace_last_linear_layer(model, new_num_classes)

    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_adam_optimizer(model.parameters(), lr=args.LEARNING_RATE, wd=args.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.LEARNING_RATE, steps_per_epoch=len(trainloader), epochs=args.EPOCHS)

    train(model, criterion, optimizer, ema=None, num_epochs=30, lr_scheduler=lr_scheduler, init_lr=args.LEARNING_RATE, task='classification', trainloader=trainloader, testloader=testloader, model_path=model_path)

if __name__ == "__main__":
    main()
    
    
# Rotation ViT with EMA
def main():
    args, unknown = parser.parse_known_args()
    args.checkpoint_dir = '/content/assignment3/assignment3/assignment3_part1/Checkpoints'
    args.seed = 315

    setup_seed(args.seed)
    print_gpu_info(device)

    print("\n--- Preparing Data for Rotation Task ---\n")

    transform_train, transform_test = get_transforms()

    trainset = CIFAR10Rotation(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.BATCH_SIZE,
                                              shuffle=True, num_workers=args.NUM_WORKERS)

    testset = CIFAR10Rotation(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.BATCH_SIZE,
                                            shuffle=False, num_workers=args.NUM_WORKERS)

    print("\n--- Training ViT on Rotation Task with EMA ---\n")

    model_path = '/content/assignment3/assignment3/assignment3_part1/ViT_rotation_task_EMA_best_model.pth'
    project_name = 'cs_mp3'

    model = ViT(args.NUM_CLASSES, args.IMAGE_SIZE, channels=args.CHANNELS, head_channels=args.HEAD_CHANNELS, num_blocks=args.NUM_BLOCKS, patch_size=args.PATCH_SIZE,
               emb_p_drop=args.EMB_P_DROP, trans_p_drop=args.TRANS_P_DROP, head_p_drop=args.HEAD_P_DROP).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_adam_optimizer(model.parameters(), lr=args.LEARNING_RATE, wd=args.WEIGHT_DECAY)
    # lr_scheduler = build_scheduler(args, optimizer)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.LEARNING_RATE, steps_per_epoch=len(trainloader), epochs=args.EPOCHS)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    train(model, criterion, optimizer, ema=ema, num_epochs=args.EPOCHS, lr_scheduler=lr_scheduler, init_lr=args.LEARNING_RATE, task='rotation', trainloader=trainloader, testloader=testloader, model_path=model_path)

if __name__ == "__main__":
    main()


# Classification ViT with EMA
def main():
    args, unknown = parser.parse_known_args()
    args.checkpoint_dir = '/content/assignment3/assignment3/assignment3_part1/Checkpoints'
    args.seed = 315

    setup_seed(args.seed)
    print_gpu_info(device)

    print("\n--- Preparing Data for Classification Task ---\n")

    transform_train, transform_test = get_transforms()

    trainset = CIFAR10Rotation(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.BATCH_SIZE,
                                              shuffle=True, num_workers=args.NUM_WORKERS)

    testset = CIFAR10Rotation(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.BATCH_SIZE,
                                            shuffle=False, num_workers=args.NUM_WORKERS)

    print("\n--- Training ViT on Classification Task ---\n")

    path = '/content/assignment3/assignment3/assignment3_part1/ViT_rotation_task_EMA_best_model.pth'
    model_path = '/content/assignment3/assignment3/assignment3_part1/ViT_classification_task_EMA_best_model.pth'
    project_name = 'cs_mp3'

    def replace_last_linear_layer(model, num_classes):
        for name, module in reversed(model._modules.items()):
            if hasattr(module, 'children') and list(module.children()):
                replace_last_linear_layer(module, num_classes)
                return
            if isinstance(module, nn.Linear):
                num_ftrs = module.in_features
                setattr(model, name, nn.Linear(num_ftrs, num_classes))
                print(f"Replaced Linear Layer: {name}")
                return

    model = ViT(args.NUM_CLASSES, args.IMAGE_SIZE, channels=args.CHANNELS, head_channels=args.HEAD_CHANNELS, num_blocks=args.NUM_BLOCKS, patch_size=args.PATCH_SIZE,
               emb_p_drop=args.EMB_P_DROP, trans_p_drop=args.TRANS_P_DROP, head_p_drop=args.HEAD_P_DROP).to(device)

    model.load_state_dict(torch.load(path), strict=False)

    new_num_classes = 10
    replace_last_linear_layer(model, new_num_classes)

    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_adam_optimizer(model.parameters(), lr=args.LEARNING_RATE, wd=args.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.LEARNING_RATE, steps_per_epoch=len(trainloader), epochs=args.EPOCHS)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    train(model, criterion, optimizer, ema=ema, num_epochs=30, lr_scheduler=lr_scheduler, init_lr=args.LEARNING_RATE, task='classification', trainloader=trainloader, testloader=testloader, model_path=model_path)

if __name__ == "__main__":
    main()