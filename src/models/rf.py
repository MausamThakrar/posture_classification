from sklearn.ensemble import RandomForestClassifier

def build_rf():
    """
    Build a Random Forest classifier.
    """
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
