import time
import torch
import wandb
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


def eval_epoch(net, testloader, criterion, task, scaler, ema=None):
    if ema:
        with ema.average_parameters():
            net.eval()
    else:
        net.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, images_rotated, labels, cls_labels in testloader:
            if task == 'rotation':
                images, labels = images_rotated.to(device), labels.to(device)
            elif task == 'classification':
                images, labels = images.to(device), cls_labels.to(device)

            with autocast():
                outputs = net(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = total_loss / len(testloader)
    accuracy = 100 * correct / total

    log_prefix = "EMA " if ema else ""
    wandb.log({f"{log_prefix}Test Loss": avg_test_loss, f"{log_prefix}Test Accuracy": accuracy})

    print(f'TESTING{" WITH EMA" if ema else ""}:')
    print(f'Accuracy of the network on the test images: {accuracy:.2f} %')
    print(f'Average loss on the test images: {avg_test_loss:.3f}')

    return avg_test_loss, accuracy

def train(net, criterion, optimizer, num_epochs, lr_scheduler, init_lr, task, trainloader, testloader, model_path, ema=None):
    lowest_loss = float('inf')
    best_accuracy = 0.0
    scaler = GradScaler()

    wandb.init(project="cs_mp3", entity="jdeklerk10")

    print(f'TRAINING{" WITH EMA" if ema else " WITHOUT EMA"}:')

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_correct = 0.0
        running_total = 0.0
        start_time = time.time()

        net.train()
        for i, (imgs, imgs_rotated, rotation_label, cls_label) in enumerate(tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'), 0):

            if task == 'rotation':
                inputs, labels = imgs_rotated.to(device), rotation_label.to(device)
            elif task == 'classification':
                inputs, labels = imgs.to(device), cls_label.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)

            if ema:
                ema.update()
            scaler.update()

            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            _, predicted = torch.max(outputs.data, 1)
            running_loss += loss.item()
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(trainloader)
        avg_acc = 100 * running_correct / running_total

        wandb.log({"Epoch": epoch + 1, "Loss": avg_loss, "Accuracy": avg_acc, "Learning Rate": current_lr})

        print(f'\nEpoch {epoch+1} finished: Avg. Loss: {avg_loss:.3f}, Avg. Acc.: {avg_acc:.2f}%, Time: {time.time() - start_time:.2f}s')

        test_loss, test_accuracy = eval_epoch(net, testloader, criterion, task, scaler, ema)
        if test_loss < lowest_loss:
            lowest_loss = test_loss
            best_accuracy = test_accuracy
            torch.save(net.state_dict(), model_path)

            wandb.log({"Best Test Loss": lowest_loss, "Best Accuracy": best_accuracy})
        print(f'New lowest test loss: {lowest_loss:.4f} with accuracy: {best_accuracy:.2f}%')

    print(f'Finished Training. Best Test Loss: {lowest_loss:.4f}, Best Accuracy: {best_accuracy:.2f}%')
    wandb.finish()