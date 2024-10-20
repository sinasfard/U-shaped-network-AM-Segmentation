from torch.utils.data import Dataset, DataLoader, random_split
# import torchvision.transforms.v2 as v2
# import torchvision.transforms.v2.functional as tv_tensors

# Define transformations
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
        return img, masks, image_name

    def __len__(self):
        return len(self.imgs)
