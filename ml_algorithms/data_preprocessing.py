"""
Shared data loading / preprocessing for the Alzheimer dataset-manifest ML benchmark.

NOTE ON THE DATA: Alzheimer_Dataset_Details.csv is an image *manifest*, not a
clinical feature table -- it records one row per MRI slice with its file
properties (width, height, file size, ...) and class label (Disease). Width,
Height, Channels, Color Mode and Extension are constant across every row in
this dataset, so VarianceThreshold drops them automatically. The only
genuinely variable fields are "File Size (KB)" and a numeric index parsed out
of the filename -- both are weak, noisy proxies for the real label (which is
actually determined by pixel content, not file metadata). Every algorithm in
this benchmark is trained on the exact same engineered features and the exact
same train/test split, so the comparison across algorithms is fair -- but
expect accuracy for all of them to sit close to the majority-class baseline.
"""
from pathlib import Path
import re

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = Path(r"C:\Users\2421311\Downloads\archive\Alzheimer_Dataset_Details.csv")
RESULTS_DIR = Path(__file__).parent / "results"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLUMNS = ["File Size (KB)", "Width", "Height", "Channels", "Filename_Index"]


def _extract_filename_index(filename):
    match = re.search(r"(\d+)", str(filename))
    return int(match.group(1)) if match else -1


def load_and_preprocess_data(data_path=DATA_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Load the CSV, engineer features, and return a ready-to-train split.

    Returns
    -------
    X_train, X_test : np.ndarray (scaled, zero-variance columns removed)
    y_train, y_test : np.ndarray (label-encoded Disease class)
    label_names : list[str] mapping encoded ints back to class names
    """
    df = pd.read_csv(data_path)
    df["Filename_Index"] = df["Filename"].apply(_extract_filename_index)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_raw = df["Disease"].to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    variance_filter = VarianceThreshold(threshold=0.0)
    X_train = variance_filter.fit_transform(X_train)
    X_test = variance_filter.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, list(label_encoder.classes_)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, label_names = load_and_preprocess_data()
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Classes: {label_names}")
    print(f"Train class balance: {[(c, int((y_train == i).sum())) for i, c in enumerate(label_names)]}")
