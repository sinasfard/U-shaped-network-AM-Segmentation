from torch.utils.data import ConcatDataset
from sklearn.model_selection import KFold
import sklearn.metrics as metrics

def concat_data():
   return ConcatDataset([train_dataset, val_dataset])
