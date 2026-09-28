import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error

from dataset_generator import generate_flight_weather_dataset
from preprocessor import FlightDataPreprocessor
from model import FlightDelayMultiTaskNN

def train_model(epochs: int = 35, batch_size: int = 64, lr: float = 0.001):
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    data_path = 'data/flight_weather_dataset.csv'
    if not os.path.exists(data_path):
        print("Generating synthetic dataset...")
        df = generate_flight_weather_dataset(n_samples=5000)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    preprocessor = FlightDataPreprocessor()
    X = preprocessor.fit_transform(df)
    preprocessor.save('models/preprocessors.joblib')

    y_cls = df['is_delayed'].values.astype(np.float32)
    y_reg = df['delay_minutes'].values.astype(np.float32)

    # Split dataset
    X_train_val, X_test, y_cls_tv, y_cls_test, y_reg_tv, y_reg_test = train_test_split(
        X, y_cls, y_reg, test_size=0.1, random_state=42
    )
    X_train, X_val, y_cls_train, y_cls_val, y_reg_train, y_reg_val = train_test_split(
        X_train_val, y_cls_tv, y_reg_tv, test_size=0.1111, random_state=42 # 0.1111 * 0.9 ~ 0.1
    )

    # Create PyTorch DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_cls_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_reg_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_cls_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_reg_val, dtype=torch.float32).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_cls_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_reg_test, dtype=torch.float32).unsqueeze(1)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X.shape[1]
    model = FlightDelayMultiTaskNN(input_dim=input_dim)

    bce_loss_fn = nn.BCELoss()
    reg_loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {'train_loss': [], 'val_loss': []}

    print("Training PyTorch Flight Delay Neural Network...")
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for b_x, b_y_cls, b_y_reg in train_loader:
            optimizer.zero_grad()
            pred_prob, pred_min = model(b_x)
            
            loss_cls = bce_loss_fn(pred_prob, b_y_cls)
            loss_reg = reg_loss_fn(pred_min, b_y_reg) / 50.0 # Scale regression loss
            
            total_loss = loss_cls + loss_reg
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item() * len(b_x)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_x, b_y_cls, b_y_reg in val_loader:
                pred_prob, pred_min = model(b_x)
                loss_cls = bce_loss_fn(pred_prob, b_y_cls)
                loss_reg = reg_loss_fn(pred_min, b_y_reg) / 50.0
                v_loss = loss_cls + loss_reg
                val_loss += v_loss.item() * len(b_x)

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        history['train_loss'].append(round(train_loss, 4))
        history['val_loss'].append(round(val_loss, 4))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/flight_delay_nn.pt')

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Evaluate on Test Set
    model.load_state_dict(torch.load('models/flight_delay_nn.pt'))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        test_probs, test_mins = model(X_test_tensor)

    test_probs_np = test_probs.numpy().ravel()
    test_mins_np = test_mins.numpy().ravel()
    test_preds_binary = (test_probs_np > 0.5).astype(int)

    acc = float(accuracy_score(y_cls_test, test_preds_binary))
    prec = float(precision_score(y_cls_test, test_preds_binary, zero_division=0))
    rec = float(recall_score(y_cls_test, test_preds_binary, zero_division=0))
    f1 = float(f1_score(y_cls_test, test_preds_binary, zero_division=0))
    auc = float(roc_auc_score(y_cls_test, test_probs_np))
    mae = float(mean_absolute_error(y_reg_test, test_mins_np))
    rmse = float(np.sqrt(mean_squared_error(y_reg_test, test_mins_np)))

    # Compute Feature Importance via perturbation
    baseline_auc = auc
    feature_importance = {}
    for i, feature_name in enumerate(preprocessor.feature_names):
        X_permuted = X_test.copy()
        np.random.shuffle(X_permuted[:, i])
        with torch.no_grad():
            perm_probs, _ = model(torch.tensor(X_permuted, dtype=torch.float32))
        perm_auc = roc_auc_score(y_cls_test, perm_probs.numpy().ravel())
        drop = max(0.0, baseline_auc - perm_auc)
        feature_importance[feature_name] = round(float(drop), 4)

    # Sort feature importance
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])

    metrics = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(auc, 4),
        'mae_minutes': round(mae, 2),
        'rmse_minutes': round(rmse, 2),
        'train_history': history,
        'feature_importance': sorted_importance,
        'dataset_size': len(df)
    }

    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel training complete!")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | MAE: {mae:.2f} mins")

if __name__ == '__main__':
    train_model()
