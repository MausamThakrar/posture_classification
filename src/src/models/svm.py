from sklearn.svm import SVC

def build_svm():
    """
    Build an RBF-kernel SVM classifier.
    """
    return SVC(kernel="rbf", probability=True, random_state=42)
