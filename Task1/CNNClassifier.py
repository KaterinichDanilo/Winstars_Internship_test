from MnistClassifierInterface import MnistClassifierInterface
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class CNNClassifier(MnistClassifierInterface, nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Dropout2d(0.2)
        )

        self.linear_layers = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features=9216, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=10)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    'min', patience=3, factor=0.5)
        self.model_type = "Convolutional Neural Network"

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.linear_layers(x)
        return x

    @staticmethod
    def reshape_data(x):
        x_array = np.array(x, dtype=np.float32)

        if x_array.ndim == 2 and x_array.shape[1] == 784: # (N, 784)
            x_array = x_array.reshape(-1, 1, 28, 28)
        elif x_array.ndim == 3:  # (N, 28, 28)
            x_array = x_array.reshape(-1, 1, 28, 28)

        return torch.from_numpy(x_array / 255.0).float()

    def train(self, x_train, y_train):
        nn.Module.train(self, True)
        x_tensor = CNNClassifier.reshape_data(x_train)
        y_tensor = torch.tensor(np.array(y_train), dtype=torch.int64)
        train_dataset = TensorDataset(x_tensor, y_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

        epochs = 15

        for epoch in range(epochs):
            running_loss = 0.
            epoch_loss = 0.

            for i, (data, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                predictions = self.forward(data)
                loss = self.loss_fn(predictions, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                epoch_loss += loss.item()

                if i % 100 == 99:
                    last_loss = running_loss / 100  # loss per batch
                    print('epoch {}  batch {} loss: {}'.format(epoch + 1, i + 1, last_loss))
                    running_loss = 0.

            avg_epoch_loss = epoch_loss / len(train_loader)
            self.scheduler.step(avg_epoch_loss)

    def predict(self, x_test):
        nn.Module.train(self, False)
        x_tensor = CNNClassifier.reshape_data(x_test)

        if len(x_tensor) > 500:
            test_dataset = TensorDataset(x_tensor)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            all_predictions = []

            with torch.no_grad():
                for batch in test_loader:
                    outputs = self.forward(batch[0])
                    _, predicted = torch.max(outputs, 1)
                    all_predictions.append(predicted.numpy())

            return np.concatenate(all_predictions)

        with torch.no_grad():
            outputs = self.forward(x_tensor)
            _, predicted = torch.max(outputs, 1)

            return predicted.numpy()