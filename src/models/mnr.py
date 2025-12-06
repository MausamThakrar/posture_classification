from sklearn.linear_model import LogisticRegression

def build_mnr():
    """
    Build a multinomial logistic regression classifier.
    """
    return LogisticRegression(
        max_iter=1000,
        multi_class="auto",
    )
