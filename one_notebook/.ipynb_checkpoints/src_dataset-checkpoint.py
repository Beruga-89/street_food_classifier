
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.config import DATA_DIR, BATCH_SIZE, SEED
def get_data_loaders():
    transform_train = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform_val = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform_train)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    val_set.dataset.transform = transform_val
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader
