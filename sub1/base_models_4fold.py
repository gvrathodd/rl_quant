import os
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

DATA_ROOT   = "../../EBY/"
LABEL_ROOT  = "../data/label1/"
PROB_ROOT   = "../data/probs/"

os.makedirs(PROB_ROOT, exist_ok=True)

DAY_START   = 1
DAY_END     = 200
H_VALUES    = list(range(50, 2001, 50))
N_FOLDS     = 4
TRAIN_DAYS  = 150
TEST_DAYS   = 50

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

def load_day_labels(day_index, col_name):
    df = pd.read_csv(f"{LABEL_ROOT}day{day_index}.csv")
    labels = df[col_name]
    return labels

for h in H_VALUES:
    col_name = f"class_h{h}"
    print(f"\n=== Processing label column: {col_name} ===")

    fold_splits = [
        (list(range(1, 151)), list(range(151, 201))),
        (list(range(1, 101)) + list(range(151, 201)), list(range(101, 151))),
        (list(range(1, 51)) + list(range(101, 201)), list(range(51, 101))),
        (list(range(51, 201)), list(range(1, 51)))
    ]

    for fold_i, (train_days, test_days) in enumerate(fold_splits, 1):
        print(f"\n--- Fold {fold_i}: Training on {len(train_days)} days, predicting {len(test_days)} days ---")

        X_train_list, y_train_list = [], []
        for d in train_days:
            Xd, _ = load_day_data(d)
            yd = load_day_labels(d, col_name)
            valid_idx = ~yd.isna()
            X_train_list.append(Xd.loc[valid_idx])
            y_train_list.append(yd.loc[valid_idx])
        X_train = pd.concat(X_train_list, ignore_index=True)
        y_train = pd.concat(y_train_list, ignore_index=True)

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=5,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        model.fit(X_train, y_train_enc)
        print(f"Model trained for {col_name}, fold {fold_i}")

        for d in test_days:
            Xd, price = load_day_data(d)
            yd = load_day_labels(d, col_name)
            valid_idx = ~yd.isna()

            Xd_valid = Xd.loc[valid_idx]
            price_valid = price.loc[valid_idx].reset_index(drop=True)

            probs = model.predict_proba(Xd_valid)
            prob_df = pd.DataFrame(probs, columns=[f"prob_{col_name}_class{c}" for c in range(5)])

            out_df = pd.concat([price_valid, prob_df], axis=1)
            out_path = f"{PROB_ROOT}day{d}.csv"

            if os.path.exists(out_path):
                existing = pd.read_csv(out_path)
                combined = pd.concat([existing, prob_df], axis=1)
                combined.to_csv(out_path, index=False)
            else:
                out_df.to_csv(out_path, index=False)

            print(f"Saved probabilities for {col_name}, fold {fold_i}, day{d}")

print("\n=== All label columns processed and probabilities saved ===")
