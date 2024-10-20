
# Set seed for reproducibility
torch.manual_seed(42)

dataset_root = r"C:\AM-Project\Data4\Data"

full_dataset = CustomDataset(root=dataset_root)

# Define dataset sizes
train_ratio, val_ratio, test_ratio = 0.75, 0.20, 0.05
dataset_size = len(full_dataset)
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)


train_dataset.dataset = CustomDataset(root=dataset_root, transforms=transform_train)
val_dataset.dataset = CustomDataset(root=dataset_root, transforms=transform_test)
test_dataset.dataset = CustomDataset(root=dataset_root, transforms=transform_test)



# Define DataLoaders
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
      loss_eval.update(loss.item(), weight=len(targets))
      metric(outputs, targets)
  return loss_eval.compute().item(), metric.compute().item()
