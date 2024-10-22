from torch.utils.data import ConcatDataset
from sklearn.model_selection import KFold
import sklearn.metrics as metrics

def train_kfold(num_epochs, k_folds):
      dataset = ConcatDataset([train_dataset, val_dataset])
      
      num_epochs = num_epochs
      k_folds = k_folds
      kfold = KFold(n_splits=k_folds, shuffle=True)
      
      # Start print
      print('--------------------------------')
      # K-fold Cross Validation model evaluation
      for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
      
      # Print
          print(f'FOLD {fold}')
          print('--------------------------------')
          
          # Sample elements randomly from a given list of ids, no replacement.
          train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
          test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
          
          # Define data loaders for training and testing data in this fold
          train_loader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size=10, sampler=train_subsampler)
          val_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=10, sampler=test_subsampler)
          
              
              # Run the training loop for defined number of epochs
          for epoch in range(num_epochs):
        # Train
            model, loss_train, metric_train= train_one_epoch(model,
                                                              train_loader,
                                                              loss_fn,
                                                              optimizer,
                                                              metric,
                                                              epoch+1)
            # Validation
            loss_valid, metric_valid, miou, mdice = evaluate(model,
                                                val_loader,
                                                loss_fn,
                                                metric)
      
            loss_train_hist.append(loss_train)
            loss_valid_hist.append(loss_valid)
      
            metric_train_hist.append(metric_train)
            # metric_pre_hist.append(metric_pre)
            metric_valid_hist.append(metric_valid)
      
            if loss_valid < best_loss_valid:
              torch.save(model, f'U_transform_kfold.pt')
              best_loss_valid = loss_valid
              print('Model Saved!')
      
            if wandb_enable:
              wandb.log({"metric_train": metric_train, "loss_train": loss_train,
                          "metric_valid": metric_valid, "loss_valid": loss_valid})
                
            # conv = cfs(predictions, labels.type(torch.uint8))
            # miou,mdice,Acc,Se,Sp,IU,f1 = calculate_Accuracy(conv.cpu())
      
            print(f'Valid: Loss = {loss_valid:.4}, Metric = {metric_valid:.4}, mIoU = {miou:.4}, mDice = {mdice:.4}, LR = {lr_scheduler.get_last_lr()[0]}')
            print()
      
            lr_scheduler.step()
      
            epoch_counter += 1
