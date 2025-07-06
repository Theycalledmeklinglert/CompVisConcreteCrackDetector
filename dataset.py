from typing import List, Optional, Sequence, Union

from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision import transforms, datasets


class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


def pil_loader(path):
    return Image.open(path).convert("RGB")


class LabeledWrapper(Dataset):
    def __init__(self, dataset, label: Union[int, float]):
        self.dataset = dataset
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]  # discard original label
        return x, self.label



class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        train_data_path: str,
        test_data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_data_dir = train_data_path
        self.test_data_dir = test_data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers


    def setup(self, stage: Optional[str] = None) -> None:

        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              #transforms.CenterCrop(148),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                               ])
        
        # val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                     transforms.CenterCrop(148),
        #                                     transforms.Resize(self.patch_size),
        #                                     transforms.ToTensor(),])

        # ---------- load all NON-cracked images ----------------------------------
        full_noncrack = datasets.DatasetFolder(
            self.train_data_dir,
            loader=pil_loader,
            extensions=("jpg", "jpeg", "png"),
            transform=train_transforms,
        )

        # ---------- 65 / 20 / 15 split ------------------------------------------
        n_total = len(full_noncrack)
        n_train = int(0.65 * n_total)
        n_val = int(0.20 * n_total)
        n_test_nc = n_total - n_train - n_val

        train_nc, val_nc, test_nc = random_split(
            full_noncrack,
            [n_train, n_val, n_test_nc],
            # generator=torch.Generator().manual_seed(42)
        )

        # ---------- wrap with fixed labels --------------------------------------
        self.train_dataset = LabeledWrapper(train_nc, label=0.0)  # non-crack
        self.val_dataset = LabeledWrapper(val_nc, label=0.0)  # non-crack
        test_normal = LabeledWrapper(test_nc, label=0.0)  # non-crack

        # ---------- load cracked images (label = 1) ------------------------------
        full_cracked_ds = datasets.DatasetFolder(
            self.test_data_dir,
            loader=pil_loader,
            extensions=("jpg", "jpeg", "png"),
            transform=train_transforms,
        )

        n_cracked_total = len(full_cracked_ds)
        c_test_size = int(0.15 * n_cracked_total)
        c_rest_size = n_cracked_total - c_test_size

        # print("c_test_size", c_test_size)
        # print("c_rest_size", c_rest_size)
        # print("n_test_nc", n_test_nc)

        cracked_ds, _ = random_split(
            full_cracked_ds,
            [c_test_size, c_rest_size],
            # generator=torch.Generator().manual_seed(42)
        )

        cracked_ds = LabeledWrapper(cracked_ds, label=1.0)

        # print("cracked_ds length:", len(cracked_ds))
        # print("nc length:", len(test_normal))

        # ---------- final test set:  normal  + cracked ---------------------------
        self.test_dataset = ConcatDataset([test_normal, cracked_ds])

#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
     