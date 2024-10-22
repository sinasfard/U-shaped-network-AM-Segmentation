import argparse
import Metrics
# from k_fold import *
import AM_network
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms as v2
from sklearn.model_selection import KFold
import tqdm
from torchmetrics import ConfusionMatrix

# Argument Parser Setup
parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs of training")
parser.add_argument("--k_fold", type=int, default=4, help="Number of k-folds for cross-validation")
parser.add_argument("--data_root_path", type=str, default='/content/data', help="Dataset root path")

arg = parser.parse_args()
# print('hi')
# print(f"Num Epochs: {arg.num_epochs}, KFold: {arg.kfold}, Data Root Path: {arg.data_root_path}")

# Device setup
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Optimizer configuration (assuming model will be passed later)
lr = 0.00003
wd = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Define the training function
def train(root, num_epochs, k_fold):
    
    # Data augmentation for training and testing
    transform_train = v2.Compose([
        v2.CenterCrop((300, 380)),
        v2.Resize(size=(224, 224), antialias=True),
        v2.RandomPhotometricDistort(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
        v2.Normalize(mean=(0.5,), std=(0.5,)),
        v2.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

    transform_test = v2.Compose([
        v2.CenterCrop((300, 380)),
        v2.Resize(size=(224, 224), antialias=True),
        v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
        v2.Normalize(mean=(0.5,), std=(0.5,)),
        v2.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Load datasets
    train_dataset = CustomDataset(root=root, transforms=transform_train)

    # Split dataset into train, validation, and test sets
    dataset_size = len(train_dataset)
    train_size = int(0.75 * dataset_size)
    val_size = int(0.20 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_size, val_size, test_size])
    test_dataset.dataset = CustomDataset(root=root, transforms=transform_test)

    # DataLoader setup
    batch_size = 32
    num_workers = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    # Initialize model, optimizer, and loss function
    model = AM_network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[350], gamma=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    # K-Fold Cross-Validation
    def train_kfold(num_epochs, k_fold):
        kfold1 = KFold(n_splits=k_fold, shuffle=True)

        print('Starting K-Fold Cross-Validation')
        for fold, (train_ids, test_ids) in enumerate(kfold1.split(train_dataset)):
            print(f'Fold {fold + 1}/{kfold}')

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            fold_train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
            fold_val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_subsampler)

            best_loss = float('inf')

            for epoch in range(num_epochs):
                # Train the model
                model, loss_train, metric_train = train_one_epoch(
                    model, fold_train_loader, loss_fn, optimizer, epoch + 1
                )

                # Validate the model
                loss_val, metric_val, miou, mdice = evaluate(
                    model, fold_val_loader, loss_fn
                )

                # Save the model if it improves
                if loss_val < best_loss:
                    torch.save(model.state_dict(), f'model_fold_{fold}.pt')
                    best_loss = loss_val
                    print(f'Model for fold {fold + 1} saved!')

                print(f'Epoch {epoch + 1}: Train Loss={loss_train:.4f}, Val Loss={loss_val:.4f}')

            lr_scheduler.step()

    # Train with K-Fold
    train_kfold(num_epochs, k_fold)

# Define training helper functions
def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=None):
  model.train()
  loss_train = MeanMetric()
  metric.reset()

  with tqdm.tqdm(train_loader, unit='batch') as tepoch:
    for inputs, targets in tepoch:
      if epoch:
        tepoch.set_description(f'Epoch {epoch}')      
      inputs = inputs.to(device)
      targets = targets.to(device)
      targets = targets.type(torch.uint8)
      outputs= model(inputs)
      loss = loss_fn(outputs, targets)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      loss_train.update(loss.item(), weight=len(targets))
      metric.update(outputs, targets)
      tepoch.set_postfix(loss=loss_train.compute().item(),
                         metric=metric.compute().item())
  return model, loss_train.compute().item(), metric.compute().item()

def evaluate(model, test_loader, loss_fn, metric):
  model.eval()
  loss_eval = MeanMetric()
  metric.reset()
  with torch.inference_mode():
    for inputs, targets in test_loader:
      inputs = inputs.to(device)
      targets = targets.to(device)
      targets = targets.type(torch.uint8)
      outputs = model(inputs)
      loss = loss_fn(outputs, targets)
      conv = cfs(outputs, targets)#.type(torch.uint8))
      miou,mdice,Acc,Se,Sp,IU,f1 = calculate_Accuracy(conv.cpu())
      loss_eval.update(loss.item(), weight=len(targets))
      metric(outputs, targets)
  return loss_eval.compute().item(), metric.compute().item(), miou, mdice



def calculate_Accuracy(confusion):
    confusion=np.asarray(confusion)
    pos = np.sum(confusion, 1).astype(np.float32) # 1 for row
    res = np.sum(confusion, 0).astype(np.float32) # 0 for coloum
    tp = np.diag(confusion).astype(np.float32)
    f1 = 2 * confusion[1][1] / (2 * confusion[1][1] + confusion[1][0] + confusion[0][1])
    IU = tp / (pos + res - tp)
    dice = 2 * tp / (pos+res)
    meanDice = np.mean(dice)
    meanIU = np.mean(IU)
    Acc = np.sum(tp) / np.sum(confusion)
    Se = confusion[1][1] / (confusion[1][1]+confusion[0][1])
    Sp = confusion[0][0] / (confusion[0][0]+confusion[1][0])
    # Fpr = confusion[1][0]
    # Tpr = confusion[1][1]

    # return  meanIU,meanDice,Acc,Se,Sp,IU,f1
    return  IU[1],dice[1],Acc,Se,Sp,IU,f1
if __name__ == "__main__":
    train(**vars(arg))
