import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
    root_mean_squared_error,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Network.csv")

CATEGORICAL_COLS = [
    "Service_Class",
    "Network_Slice",
    "Network_Function",
    "Deployment_Zone",
    "Traffic_Direction",
    "Network_Condition",
    "Energy_Management_Status",
]

DROP_COLS = [
    "Record_ID",
    "Timestamp",
    "Future_Traffic_Demand_Mbps",
    "Future_Congestion_Index",
]

# Features most correlated with Future_Traffic_Demand_Mbps
TRAFFIC_FEATURES = [
    "Predicted_Energy_Requirement_kWh",
    "Peak_Time",
    "Traffic_Volume_Mbps",
    "Throughput_Mbps",
    "Hour",
    "Active_User_Count",
    "VNF_Instance_Count",
    "User_Plane_Load_Pct",
    "Resource_Utilization_Efficiency_Pct",
    "Congestion_Index",
    "Downlink_Traffic_Mbps",
    "Energy_Efficiency_Mbps_per_kWh",
    "Packet_Arrival_Rate_Kpps",
    "Memory_Utilization_Pct",
    "Active_Core_Nodes",
    # Categorical columns (will be one-hot encoded)
    "Service_Class",
    "Network_Slice",
    "Network_Condition",
    "Traffic_Direction",
]

# Features most correlated with Future_Congestion_Index
CONGESTION_FEATURES = [
    "Predicted_Energy_Requirement_kWh",
    "Peak_Time",
    "Traffic_Volume_Mbps",
    "Throughput_Mbps",
    "Hour",
    "Resource_Utilization_Efficiency_Pct",
    "User_Plane_Load_Pct",
    "Active_User_Count",
    "VNF_Instance_Count",
    "Congestion_Index",
    "Packet_Arrival_Rate_Kpps",
    "Memory_Utilization_Pct",
    "Downlink_Traffic_Mbps",
    "Energy_Efficiency_Mbps_per_kWh",
    "CPU_Utilization_Pct",
    # Categorical columns (will be one-hot encoded)
    "Service_Class",
    "Network_Slice",
    "Network_Condition",
    "Network_Condition",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feat_df = df.copy()
    eps = 1e-6
    if "Traffic_Volume_Mbps" in feat_df.columns and "Bandwidth_Allocation_MHz" in feat_df.columns:
        feat_df["Traffic_to_BW"] = feat_df["Traffic_Volume_Mbps"] / (feat_df["Bandwidth_Allocation_MHz"] + eps)
    if "Throughput_Mbps" in feat_df.columns and "Traffic_Volume_Mbps" in feat_df.columns:
        feat_df["Throughput_Ratio"] = feat_df["Throughput_Mbps"] / (feat_df["Traffic_Volume_Mbps"] + eps)
    if "User_Plane_Load_Pct" in feat_df.columns and "Control_Plane_Load_Pct" in feat_df.columns:
        feat_df["Load_Product"] = feat_df["User_Plane_Load_Pct"] * feat_df["Control_Plane_Load_Pct"]
    if "Active_Core_Nodes" in feat_df.columns and "Idle_Core_Nodes" in feat_df.columns:
        feat_df["Resource_Stress"] = feat_df["Active_Core_Nodes"] / (
            feat_df["Active_Core_Nodes"] + feat_df["Idle_Core_Nodes"] + eps
        )
    if "CPU_Utilization_Pct" in feat_df.columns and "Memory_Utilization_Pct" in feat_df.columns:
        feat_df["Compute_Load"] = (feat_df["CPU_Utilization_Pct"] + feat_df["Memory_Utilization_Pct"]) / 2.0
    if "Latency_ms" in feat_df.columns and "Packet_Loss_Rate_Pct" in feat_df.columns:
        feat_df["Network_Stress"] = feat_df["Latency_ms"] * (1.0 + feat_df["Packet_Loss_Rate_Pct"])
    return feat_df


def evaluate_regression(y_true, y_pred, target_name: str) -> dict:
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    accuracy_1_minus_mape = float((1.0 - mape) * 100.0)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mean_val = float(np.mean(y_true))
    normalized_accuracy = float((1.0 - mae / mean_val) * 100.0)

    print(f"\n==================== Evaluation: {target_name} ====================")
    print(f"  R2 Score (Variance Explained)   : {r2 * 100:.2f}%  (R2 = {r2:.4f})")
    print(f"  Regression Accuracy (1 - MAPE)  : {accuracy_1_minus_mape:.2f}% (MAPE = {mape:.4f})")
    print(f"  Normalized Accuracy (1 - MAE/y) : {normalized_accuracy:.2f}%")
    print(f"  Mean Absolute Error (MAE)       : {mae:.3f}")
    print(f"  Root Mean Squared Error (RMSE)  : {rmse:.3f}")


def predict_future(
    input_data: dict | pd.DataFrame,
    traffic_model: HistGradientBoostingRegressor,
    congestion_ridge: Ridge,
    congestion_hgb: HistGradientBoostingRegressor,
    traffic_feature_cols: list[str],
    congestion_feature_cols: list[str],
) -> dict:
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()

    input_df = engineer_features(input_df)

    # Stage 1 – select only the columns that actually exist in input_df
    traffic_cat_cols = [c for c in TRAFFIC_FEATURES if c in CATEGORICAL_COLS and c in input_df.columns]
    traffic_num_cols = [c for c in TRAFFIC_FEATURES if c not in CATEGORICAL_COLS and c in input_df.columns]

    traffic_input = input_df[traffic_num_cols + traffic_cat_cols].copy()
    traffic_input = pd.get_dummies(traffic_input, columns=traffic_cat_cols, drop_first=True)
    # Pad any missing one-hot columns (unseen features) with 0
    traffic_input = traffic_input.reindex(columns=traffic_feature_cols, fill_value=0)

    predicted_traffic = traffic_model.predict(traffic_input.values)

    # Stage 2 – same pattern, missing columns filled with 0
    congestion_cat_cols = [c for c in CONGESTION_FEATURES if c in CATEGORICAL_COLS and c in input_df.columns]
    congestion_num_cols = [c for c in CONGESTION_FEATURES if c not in CATEGORICAL_COLS and c in input_df.columns]

    congestion_input = input_df[congestion_num_cols + congestion_cat_cols].copy()
    congestion_input = pd.get_dummies(congestion_input, columns=congestion_cat_cols, drop_first=True)
    congestion_input = congestion_input.reindex(columns=congestion_feature_cols, fill_value=0)

    congestion_input_chained = np.hstack([congestion_input.values, predicted_traffic[:, np.newaxis]])

    yc_ridge = congestion_ridge.predict(congestion_input_chained)
    yc_hgb = congestion_hgb.predict(congestion_input_chained)
    predicted_congestion = 0.5 * yc_ridge + 0.5 * yc_hgb

    if isinstance(input_data, dict):
        return {
            "Future_Traffic_Demand_Mbps": float(predicted_traffic[0]),
            "Future_Congestion_Index": float(predicted_congestion[0]),
        }

    return {
        "Future_Traffic_Demand_Mbps": predicted_traffic,
        "Future_Congestion_Index": predicted_congestion,
    }


def train_models():
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape: {df.shape} (Rows: {len(df)}, Columns: {len(df.columns)})")

    df_engineered = engineer_features(df)
    feature_df = df_engineered.drop(columns=DROP_COLS)
    feature_df_encoded = pd.get_dummies(feature_df, columns=CATEGORICAL_COLS, drop_first=True)

    feature_columns = feature_df_encoded.columns.tolist()
    print(f"Prepared {len(feature_columns)} training features.")

    y_traffic = df["Future_Traffic_Demand_Mbps"].values
    y_congestion = df["Future_Congestion_Index"].values
    X = feature_df_encoded.values

    X_train, X_test, yt_train, yt_test, yc_train, yc_test = train_test_split(
        X, y_traffic, y_congestion, test_size=0.2, random_state=42
    )

    t_cat = [c for c in TRAFFIC_FEATURES if c in CATEGORICAL_COLS]
    t_num = [c for c in TRAFFIC_FEATURES if c not in CATEGORICAL_COLS and c in df_engineered.columns]
    traffic_subset = pd.get_dummies(df_engineered[t_num + t_cat], columns=t_cat, drop_first=True)
    traffic_feature_cols = traffic_subset.columns.tolist()
    print(f"Stage-1 feature subset  : {len(traffic_feature_cols)} columns")

    c_cat = list(dict.fromkeys([c for c in CONGESTION_FEATURES if c in CATEGORICAL_COLS]))  # deduplicated
    c_num = [c for c in CONGESTION_FEATURES if c not in CATEGORICAL_COLS and c in df_engineered.columns]
    congestion_subset = pd.get_dummies(df_engineered[c_num + c_cat], columns=c_cat, drop_first=True)
    congestion_feature_cols = congestion_subset.columns.tolist()
    print(f"Stage-2 feature subset  : {len(congestion_feature_cols)} columns")

    Xt_train, Xt_test, _, _ = train_test_split(
        traffic_subset.values, y_traffic, test_size=0.2, random_state=42
    )
    Xc_train, Xc_test, _, _ = train_test_split(
        congestion_subset.values, y_congestion, test_size=0.2, random_state=42
    )
    print("\nTraining Stage 1 Model: Future_Traffic_Demand_Mbps...")
    traffic_model = HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.07,
        min_samples_leaf=20,
        random_state=42,
    )
    traffic_model.fit(X_train, yt_train)
    yt_pred = traffic_model.predict(X_test)
    evaluate_regression(yt_test, yt_pred, "Future_Traffic_Demand_Mbps")

    print("  Training prediction model on feature subset...")
    traffic_pred_model = HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.07,
        min_samples_leaf=20,
        random_state=42,
    )
    traffic_pred_model.fit(Xt_train, yt_train)

    print("\nTraining Stage 2 Model: Future_Congestion_Index (Chained with predicted traffic)...")
    pt_train = traffic_model.predict(X_train)[:, np.newaxis]
    pt_test = yt_pred[:, np.newaxis]

    X_train_c = np.hstack([X_train, pt_train])
    X_test_c = np.hstack([X_test, pt_test])

    ridge_m = Ridge(alpha=10.0)
    ridge_m.fit(X_train_c, yc_train)

    hgb_m = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=350,
        learning_rate=0.08,
        min_samples_leaf=20,
        random_state=42,
    )
    hgb_m.fit(X_train_c, yc_train)

    yc_pred_ridge = ridge_m.predict(X_test_c)
    yc_pred_hgb = hgb_m.predict(X_test_c)
    yc_pred = 0.5 * yc_pred_ridge + 0.5 * yc_pred_hgb
    evaluate_regression(yc_test, yc_pred, "Future_Congestion_Index")

    print("  Training prediction model on feature subset...")
    pt_sub_train = traffic_pred_model.predict(Xt_train)[:, np.newaxis]
    pt_sub_test = traffic_pred_model.predict(Xt_test)[:, np.newaxis]

    Xc_train_chained = np.hstack([Xc_train, pt_sub_train])
    Xc_test_chained = np.hstack([Xc_test, pt_sub_test])

    congestion_pred_ridge = Ridge(alpha=10.0)
    congestion_pred_ridge.fit(Xc_train_chained, yc_train)

    congestion_pred_hgb = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=350,
        learning_rate=0.08,
        min_samples_leaf=20,
        random_state=42,
    )
    congestion_pred_hgb.fit(Xc_train_chained, yc_train)

    return {
        "traffic_model": traffic_pred_model,
        "congestion_ridge": congestion_pred_ridge,
        "congestion_hgb": congestion_pred_hgb,
        "traffic_feature_cols": traffic_feature_cols,
        "congestion_feature_cols": congestion_feature_cols,
    }


if __name__ == "__main__":
    artifacts = train_models()
    print("\n\n========== Predict Future Network State ==========")
    print("Enter the following 5 network features:\n")

    traffic_volume  = float(input("  Traffic_Volume_Mbps   : "))
    active_users    = float(input("  Active_User_Count     : "))
    congestion_idx  = float(input("  Congestion_Index      : "))
    uplink_traffic  = float(input("  Uplink_Traffic_Mbps   : "))
    downlink_traffic = float(input("  Downlink_Traffic_Mbps : "))

    user_input = {
        "Traffic_Volume_Mbps":    traffic_volume,
        "Active_User_Count":      active_users,
        "Congestion_Index":       congestion_idx,
        "Uplink_Traffic_Mbps":    uplink_traffic,
        "Downlink_Traffic_Mbps":  downlink_traffic,
    }

    predictions = predict_future(
        input_data=user_input,
        traffic_model=artifacts["traffic_model"],
        congestion_ridge=artifacts["congestion_ridge"],
        congestion_hgb=artifacts["congestion_hgb"],
        traffic_feature_cols=artifacts["traffic_feature_cols"],
        congestion_feature_cols=artifacts["congestion_feature_cols"],
    )

    print("\n---------- Prediction Results ----------")
    print(f"  Future Traffic Demand  : {predictions['Future_Traffic_Demand_Mbps']:.3f} Mbps")
    print(f"  Future Congestion Index: {predictions['Future_Congestion_Index']:.4f}")