from torch.utils.data import Dataset, DataLoader, random_split
# import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "annotation"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "annotation", self.masks[idx])
        img = Image.open(img_path)
        masks = Image.open(mask_path)

        img = tv_tensors.Image(img)
        masks = tv_tensors.Mask(masks)

        if self.transforms:
            img, masks = self.transforms(img, masks)
        image_name = self.imgs[idx]
        return img, masks

    def __len__(self):
        return len(self.imgs)
