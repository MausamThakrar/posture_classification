from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def plot_pr(y_true, y_prob, label: str = "class"):
    """
    Plot a precision-recall curve for a given set of probabilities.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
