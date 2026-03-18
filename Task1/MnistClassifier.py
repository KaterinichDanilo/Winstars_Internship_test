from Task1.RandomForestClassifier import RandomForestModel
from Task1.FeedforwardNeuralNetworkClassifier import NeuralNetworkClassifier
from Task1.CNNClassifier import CNNClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

class MnistClassifier():
    def __init__(self, model_type: str):
        if model_type == "rf":
            self.model = RandomForestModel()
        elif model_type == "cnn":
            self.model = CNNClassifier()
        elif model_type == "nn":
            self.model = NeuralNetworkClassifier()
        else:
            raise ValueError("Invalid model type")

    def train(self, x_train, y_train):
        print(f"{self.model.model_type} start training...")
        self.model.train(x_train, y_train)
        print(f"{self.model.model_type} training finished")

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        print(f"Evaluating model {self.model.model_type}")
        y_pred = self.model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f'Accuracy: {accuracy:.4f}\n'
              f'Precision: {precision:.4f}\n'
              f'Recall: {recall:.4f}\n'
              f'F1 score: {f1:.4f}')

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model.model_type}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()