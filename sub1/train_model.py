import os
import glob
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

DATA_ROOT  = "../../EBY/"
LABEL_ROOT = "../data/label1/"
MODEL_ROOT = "../data/models2/"
PROB_ROOT  = "../data/probs_pred/"

os.makedirs(MODEL_ROOT, exist_ok=True)
os.makedirs(PROB_ROOT, exist_ok=True)

TRAIN_START, TRAIN_END = 1, 200
TEST_START, TEST_END = 201, 228

def load_day_data(day_index):
    df = pd.read_csv(f"{DATA_ROOT}day{day_index}.csv")
    df = df.iloc[:, 1:218]
    df = df.drop(df.columns[71], axis=1)
    df = df.drop(df.columns[71], axis=1)
    df = df.drop(df.columns[24], axis=1)
    df = df.dropna()
    price = df["Price"].reset_index(drop=True)
    features = df.drop(columns=["Price"]).reset_index(drop=True)
    return features, price

def load_day_labels(day_index):
    return pd.read_csv(f"{LABEL_ROOT}day{day_index}.csv")

label_example = load_day_labels(1)
label_columns = list(label_example.columns)
print(f"Detected {len(label_columns)} label columns: {label_columns[:5]} ...")

for col_name in label_columns:
    print(f"\n=== Training model for {col_name} ===")

    X_train_list, y_train_list = [], []
    for d in range(TRAIN_START, TRAIN_END + 1):
        Xd, _ = load_day_data(d)
        yd = load_day_labels(d)[col_name]
        valid_idx = ~yd.isna()
        X_train_list.append(Xd.loc[valid_idx])
        y_train_list.append(yd.loc[valid_idx])

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    print(f"Train shape for {col_name}: {X_train.shape}, labels: {y_train.shape}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        verbosity=1
    )
    model.fit(X_train, y_enc)

    model_path = os.path.join(MODEL_ROOT, f"{col_name}.json")
    model.save_model(model_path)
    print(f" Model saved to {model_path}")

    for d in range(TEST_START, TEST_END + 1):
        Xd, price = load_day_data(d)
        yd = load_day_labels(d)[col_name]
        valid_idx = ~yd.isna()

        Xd_valid = Xd.loc[valid_idx]
        price_valid = price.loc[valid_idx].reset_index(drop=True)

        probs = model.predict_proba(Xd_valid)
        prob_df = pd.DataFrame(probs, columns=[f"prob_{col_name}_class{c}" for c in range(probs.shape[1])])
        out_df = pd.concat([price_valid, prob_df], axis=1)

        out_path = os.path.join(PROB_ROOT, f"day{d}.csv")
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            combined = pd.concat([existing, prob_df], axis=1)
            combined.to_csv(out_path, index=False)
        else:
            out_df.to_csv(out_path, index=False)

        print(f"Saved probabilities for {col_name}, day{d}")

print("\n All models trained and probabilities saved successfully.")
