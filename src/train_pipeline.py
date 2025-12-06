import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from src.preprocessing.noise_reduction import reduce_noise
from src.preprocessing.dbscan_outlier_removal import remove_outliers_dbscan

from src.models.svm import build_svm
from src.models.rf import build_rf
from src.models.mnr import build_mnr
from src.models.cnn import build_cnn
from src.models.rnn import build_rnn

def load_data() -> pd.DataFrame:
    path = Path("data/sample_posture_data.csv")
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path.resolve()}")
    df = pd.read_csv(path)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = reduce_noise(df)
    df = remove_outliers_dbscan(df)
    return df

def split_scale_encode(df: pd.DataFrame):
    X = df.drop(columns=["label"])
    y = df["label"]

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc,
    )

    return X_train, X_test, y_train, y_test, scaler, encoder

def train_classical_models(X_train, X_test, y_train, y_test):
    models = {
        "SVM": build_svm(),
        "RandomForest": build_rf(),
        "MNR": build_mnr(),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        trained_models[name] = model

        print(f"{name} Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred))

    return results, trained_models

def train_deep_learning(X_train, X_test, y_train, y_test, num_classes):
    input_shape = (X_train.shape[1],)

    print("\nTraining CNN...")
    cnn = build_cnn(input_shape, num_classes)
    history_cnn = cnn.fit(
        X_train,
        y_train,
        epochs=10,
        validation_split=0.2,
        verbose=0,
    )
    cnn_loss, cnn_acc = cnn.evaluate(X_test, y_test, verbose=0)
    print(f"CNN Test Accuracy: {cnn_acc:.3f}")

    print("\nTraining RNN...")
    rnn = build_rnn(input_shape, num_classes)
    history_rnn = rnn.fit(
        X_train,
        y_train,
        epochs=10,
        validation_split=0.2,
        verbose=0,
    )
    rnn_loss, rnn_acc = rnn.evaluate(X_test, y_test, verbose=0)
    print(f"RNN Test Accuracy: {rnn_acc:.3f}")

    return {"CNN": cnn_acc, "RNN": rnn_acc}

def main():
    df = load_data()
    print("Initial data shape:", df.shape)
    df = preprocess(df)
    print("After preprocessing shape:", df.shape)

    X_train, X_test, y_train, y_test, scaler, encoder = split_scale_encode(df)
    num_classes = len(encoder.classes_)

    print("\n=== Classical ML Models ===")
    classical_results, classical_models = train_classical_models(
        X_train, X_test, y_train, y_test
    )

    print("\n=== Deep Learning Models ===")
    dl_results = train_deep_learning(X_train, X_test, y_train, y_test, num_classes)

    print("\n=== Final Results Summary ===")
    all_results = {**classical_results, **dl_results}
    for name, acc in all_results.items():
        print(f"{name}: {acc:.3f}")

    best_model_name = max(classical_results, key=classical_results.get)
    best_model = classical_models[best_model_name]
    print(f"\nBest classical model: {best_model_name}")

    joblib.dump(best_model, "best_posture_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(encoder, "label_encoder.joblib")
    print("Saved best model, scaler and encoder to disk.")

if __name__ == "__main__":
    main()
