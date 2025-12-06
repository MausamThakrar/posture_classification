import matplotlib.pyplot as plt

def plot_accuracy(history):
    """
    Plot training and validation accuracy from a Keras History object.
    """
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, label="train")
    if val_acc:
        plt.plot(epochs, val_acc, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
