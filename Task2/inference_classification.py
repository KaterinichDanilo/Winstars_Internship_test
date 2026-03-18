import argparse
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

from AnimalClassifier import AnimalClassifier

parser = argparse.ArgumentParser(description="Test animal classification model")
parser.add_argument("--data_dir", type=str, default="animals_splitted", help="Path to data")
parser.add_argument("--model_path", type=str, default="animal_resnet50.pth")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("-" * 30)
if device.type == 'cuda':
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("Training on CPU")
print("-" * 30)

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Testing data loading...")

test_dir = os.path.join(args.data_dir, 'test')
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Testing data loading finished")
print('Model loading...')

model = AnimalClassifier(device, model_path=args.model_path)

print('Model loaded')

class_names = test_dataset.classes
model.evaluate(test_loader, class_names)