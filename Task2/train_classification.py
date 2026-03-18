import argparse
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

from AnimalClassifier import AnimalClassifier

parser = argparse.ArgumentParser(description="Train Animal Classification Model")
parser.add_argument("--data_dir", type=str, default="animals_splitted", help="Path to data")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--model_path", type=str, default="animal_resnet50.pth")
parser.add_argument("--no_save", action='store_false', dest='model_save',
                    help="Don't save the model after training")
parser.set_defaults(model_save=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("-" * 30)
if device.type == 'cuda':
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("Training on CPU")
print("-" * 30)

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Стандарт ImageNet
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dir = os.path.join(args.data_dir, 'train')
val_dir = os.path.join(args.data_dir, 'val')

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

model = AnimalClassifier(device, lr=args.lr)
model.train(train_loader, val_loader, epochs=args.epochs)

if args.model_save:
    model.save_model(args.model_path)