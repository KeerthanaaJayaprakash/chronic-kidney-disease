import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Convert empty strings to NaN
    df.replace("", pd.NA, inplace=True)

    # Drop rows with missing values (simple & acceptable for this dataset)
    df.dropna(inplace=True)

    # Encode categorical features
    encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = encoder.fit_transform(df[col])

    X = df.drop("classification", axis=1)
    y = df["classification"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
