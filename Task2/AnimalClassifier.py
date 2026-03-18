import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

class AnimalClassifier:
    def __init__(self, device, model_path=None, lr=0.001):
        self.device = device

        if model_path is not None:
            self.model = AnimalClassifier.load_model(model_path, device)
        else:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            for param in self.model.parameters():
                param.requires_grad = False

            self.model.fc = nn.Linear(self.model.fc.in_features, 10)

        self.model.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=3)

    @staticmethod
    def load_model(model_path, device):
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(model_path, map_location=device))

        return model

    def train(self, train_loader, validation_loader, epochs):
        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0.0
            epoch_validation_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                labels_predict = self.model(data)
                loss = self.loss_fn(labels_predict, target)
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)

            self.model.eval()

            with torch.no_grad():
                for data, target in validation_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    predicted_vector = self.model(data)
                    loss = self.loss_fn(predicted_vector, target)
                    epoch_validation_loss += loss.item()

            avg_validation_loss = epoch_validation_loss / len(validation_loader)

            print("Epoch {}, Average train loss: {}, Average validation loss: {}, LR: {}".
                  format(epoch + 1, avg_train_loss, avg_validation_loss, self.optimizer.param_groups[0]['lr']))

            self.scheduler.step(avg_validation_loss)

    def save_model(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)
        print("Model saved: ", model_save_path)

    def predict(self, test_data):
        """
        A general-purpose method for prediction.
        :param test_data: can be a single tensor [3, 224, 224]
        or a list/batch of tensors [N, 3, 224, 224].
        :return: Return an array of indices
        """
        self.model.eval()

        if test_data.ndim == 3:
            test_data = test_data.unsqueeze(0)

        test_data = test_data.to(self.device)

        with torch.no_grad():
            outputs = self.model(test_data)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def evaluate(self, test_loader, class_names):
        self.model.eval()
        labels_predicted = []
        labels_target = []

        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)

                labels_predicted.append(predicted.cpu().numpy())
                labels_target.append(targets.numpy())

        labels_predicted = np.concatenate(labels_predicted)
        labels_target = np.concatenate(labels_target)

        accuracy = accuracy_score(labels_target, labels_predicted)
        precision = precision_score(labels_target, labels_predicted, average='weighted')
        recall = recall_score(labels_target, labels_predicted, average='weighted')
        f1 = f1_score(labels_target, labels_predicted, average='weighted')

        print(f'Accuracy: {accuracy:.4f}\n'
              f'Precision: {precision:.4f}\n'
              f'Recall: {recall:.4f}\n'
              f'F1 score: {f1:.4f}')

        print("\nClassification Report:")
        print(classification_report(labels_target, labels_predicted))

        cm = confusion_matrix(labels_target, labels_predicted)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.yticks(rotation=0, va='center')
        plt.xlabel('Predicted Label')
        plt.show()