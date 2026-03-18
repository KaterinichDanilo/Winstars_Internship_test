from MnistClassifierInterface import MnistClassifierInterface
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class NeuralNetworkClassifier(MnistClassifierInterface, nn.Module):
    def __init__(self):
        super(NeuralNetworkClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 10),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    'min', patience=3, factor=0.5)
        self.model_type = "Feed Forward Neural Network"

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def reshape_data(x):
        x_array = np.array(x).astype(float)

        # (N, 784)
        if len(x_array.shape) == 2 and x_array.shape[1] == 784:
            x_array /= 255.0

        # (N, 28, 28)
        if len(x_array.shape) == 3:
            x_array = x_array.reshape(x_array.shape[0], -1) / 255.0

        # (28, 28)
        if len(x_array.shape) == 2 and x_array.shape[0] == 28:
            x_array = x_array.reshape(1, -1) / 255.0

        return torch.from_numpy(x_array).float()

    def train(self, x_train, y_train):
        nn.Module.train(self, True)
        x_tensor = NeuralNetworkClassifier.reshape_data(x_train)
        y_tensor = torch.tensor(np.array(y_train), dtype=torch.int64)
        train_dataset = TensorDataset(x_tensor, y_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

        epochs = 20

        for epoch in range(epochs):
            running_loss = 0.
            epoch_loss = 0.

            for i, (inputs, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                prediction = self.forward(inputs)
                # Compute the loss and its gradients
                loss = self.loss_fn(prediction, labels)
                loss.backward()
                # Adjust learning weights
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
        x_tensor = NeuralNetworkClassifier.reshape_data(x_test)

        if len(x_tensor) > 500:
            test_dataset = TensorDataset(x_tensor)
            test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
            all_predictions = []

            with torch.no_grad():
                for batch in test_loader:
                    output = self.forward(batch[0])
                    _, predicted = torch.max(output, 1)
                    all_predictions.append(predicted)

            return np.concatenate(all_predictions)

        with torch.no_grad():
            output = self.forward(x_tensor)
            _, predicted = torch.max(output, 1)

        return predicted.numpy()