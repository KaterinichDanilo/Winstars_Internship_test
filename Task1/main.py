from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from MnistClassifier import MnistClassifier

if __name__ == "__main__":
    mnist = fetch_openml('mnist_784')
    X, y = mnist["data"], mnist["target"]
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    print(X_train.shape, y_train.shape)

    random_forest = MnistClassifier('rf')
    random_forest.train(X_train, y_train)
    random_forest.evaluate(X_test, y_test)

    nn = MnistClassifier('nn')
    nn.train(X_train, y_train)
    nn.evaluate(X_test, y_test)

    cnn = MnistClassifier('cnn')
    cnn.train(X_train, y_train)
    cnn.evaluate(X_test, y_test)