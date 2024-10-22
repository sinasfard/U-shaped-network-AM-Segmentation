from Metrics import calculate_Accuracy
from k_fold import train_kfold
import argparse
# Data Augmentation

parser = argparse.ArgumentParser()

parser.add_argument("--num epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("kfold", type=int, default=4, help="number of kfold")
# parser.add_argument("--n_save_iter", type=int, default=1000, help="save the model every time")
# parser.add_argument("--n_val_iter", type=int, default=1000, help="val the model every time")
# parser.add_argument("--data_root_path", type=str,
#                     default='/raid/hra/dataset/BreastCancerMRI_350/CropZero_down0.5XY/Norm/', help='dataset root path')
# parser.add_argument("--img1_prefix", type=str, default='DCE-C1', help="img prefix")
# parser.add_argument("--img2_prefix", type=str, default='DCE-C0', help="img prefix")
# parser.add_argument("--label_prefix", type=str, default='TumorMask', help="label prefix")
# parser.add_argument("--datalist_json_path", type=str, default='../data/datalist_Siemens.json', help="datalist path")
# parser.add_argument("--model_dir_root_path", type=str, default='../model/', help="root path to save the model")
# parser.add_argument("--note", type=str, default="", help="note")
arg = parser.parse_args()


transform_train = v2.Compose([
    v2.CenterCrop((300,380)),
    v2.Resize(size=(224,224), antialias=True),
    v2.RandomPhotometricDistort(p=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
    v2.Normalize(mean=(0.5,), std=(0.5,)),
    v2.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

transform_test = v2.Compose([
    v2.CenterCrop((300,380)),
    v2.Resize(size=(224,224), antialias=True),
    v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
    v2.Normalize(mean=(0.5,), std=(0.5,)),
    v2.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

# Set seed for reproducibility


torch.manual_seed(42)

dataset_root = r"C:\AM-Project\Data3\Data"

train_dataset = CustomDataset(root=dataset_root, transforms=transform_train)

train_ratio = 0.75
val_ratio = 0.20
test_ratio = 0.05

dataset_size = len(train_dataset)
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size


train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_size, val_size, test_size])
test_dataset.dataset = CustomDataset(root=dataset_root, transforms=transform_test)

your_batch_size = 32
your_num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=your_batch_size, shuffle=True, num_workers=your_num_workers)
val_loader = DataLoader(val_dataset, batch_size=your_batch_size, shuffle=False, num_workers=your_num_workers)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=your_num_workers)

#### train_one_epoch
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

####Evalution
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


def train(num_epochs, kfold):
    return train_kfold (num_epochs, kfold)


if __name__ == "__main__":

    train(**vars(arg))
