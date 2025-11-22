import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import pickle

from utils.config import RESULTS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

# Optional import: PyTorch
try:
    import torch
    from torch import nn
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# Optional import: scikit-learn
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SK_OK = True
except Exception:
    SK_OK = False


def _positional_encoding(seq_len: int, d_model: int, device) -> torch.Tensor:
    position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-np.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (seq_len, d_model)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 1, 32),  # we will use only the last token representation via masking trick
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_dim)
        batch, seq_len, in_dim = x.shape
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        # Transformer expects (seq_len, batch, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        # Add positional encoding
        pe = _positional_encoding(seq_len, self.d_model, x.device)  # (seq_len, d_model)
        x = x + pe.unsqueeze(1)  # (seq_len, batch, d_model)
        enc = self.transformer(x)  # (seq_len, batch, d_model)
        last = enc[-1]  # (batch, d_model)
        out = self.head(last.unsqueeze(1))  # (batch, 1)
        return out.squeeze(-1)


def _prepare_sequences(ts_df: pd.DataFrame, target_col: str, feature_cols: List[str], seq_len: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    df = ts_df.sort_values('year').reset_index(drop=True)
    data = df[feature_cols + [target_col]].dropna().reset_index(drop=True)
    X_list, y_list = [], []
    for i in range(seq_len, len(data)):
        X_list.append(data[feature_cols].iloc[i-seq_len:i].values)
        y_list.append(data[target_col].iloc[i])
    if not X_list:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,))
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.float32)


def run_q5_torch_transformer(results_dir: Path = RESULTS_DIR / 'q5' / 'transformer',
                              ts_df: pd.DataFrame = None,
                              target_preference: List[str] = None,
                              seq_len: int = 8,
                              epochs: int = 200,
                              lr: float = 1e-3) -> Dict[str, Any]:
    """Train a PyTorch Transformer for macro forecasting.

    - Chooses the first available target from target_preference.
    - Uses standardized multi-feature inputs.
    - Trains with early stopping and saves artifacts under results_dir.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    if not TORCH_OK or not SK_OK:
        logger.warning('PyTorch or scikit-learn not available; skipping Q5 Transformer (torch).')
        return {}

    if ts_df is None or len(ts_df) < 12:
        logger.warning('Insufficient time series for Transformer.')
        return {}

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Define target preference
    if target_preference is None:
        target_preference = ['gdp_growth', 'industrial_production', 'manufacturing_va_share']

    target_col = None
    for t in target_preference:
        if t in ts_df.columns:
            target_col = t
            break
    if target_col is None:
        logger.warning('No suitable target column found for Transformer.')
        return {}

    # Feature set (exclude target and non-numeric/year columns)
    numeric_cols = [c for c in ts_df.columns if c != target_col and c != 'year' and np.issubdtype(ts_df[c].dtype, np.number)]
    # Ensure essential drivers if present
    drivers = [c for c in ['tariff_index_total', 'retaliation_index'] if c in numeric_cols]
    feature_cols = sorted(set(numeric_cols))

    # Scale features and target
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_raw, y_raw = _prepare_sequences(ts_df, target_col, feature_cols, seq_len)
    if len(X_raw) == 0:
        logger.warning('Not enough sequential samples after preprocessing.')
        return {}

    # Fit scalers on training portion (time-ordered split)
    split = int(len(X_raw) * 0.8)
    X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
    y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

    # Flatten time dimension for scaling then reshape back
    X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    X_test_2d = X_test_raw.reshape(-1, X_test_raw.shape[-1])
    X_train_2d = scaler_x.fit_transform(X_train_2d)
    X_test_2d = scaler_x.transform(X_test_2d)
    y_train_2d = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test_2d = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

    X_train = X_train_2d.reshape(X_train_raw.shape).astype(np.float32)
    X_test = X_test_2d.reshape(X_test_raw.shape).astype(np.float32)
    y_train = y_train_2d.astype(np.float32)
    y_test = y_test_2d.astype(np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimeSeriesTransformer(in_dim=X_train.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    patience = 20
    no_improve = 0

    def to_torch(x, y):
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    Xtr, ytr = to_torch(X_train, y_train)
    Xte, yte = to_torch(X_test, y_test)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred_tr = model(Xtr)
        loss_tr = loss_fn(pred_tr, ytr)
        loss_tr.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred_te = model(Xte)
            val = loss_fn(pred_te, yte).item()

        if val < best_val - 1e-6:
            best_val = val
            no_improve = 0
            torch.save({'state_dict': model.state_dict()}, results_dir / 'q5_torch_transformer.pt')
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Load best model
    try:
        state = torch.load(results_dir / 'q5_torch_transformer.pt', map_location=device)
        model.load_state_dict(state['state_dict'])
    except Exception:
        pass

    model.eval()
    with torch.no_grad():
        pred_te = model(Xte).cpu().numpy()

    # Inverse scale to original units
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(pred_te.reshape(-1, 1)).flatten()

    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))),
        'mae': float(mean_absolute_error(y_test_orig, y_pred_orig)),
        'r2': float(r2_score(y_test_orig, y_pred_orig)),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'target': target_col,
        'seq_len': int(seq_len),
        'features_used': feature_cols,
    }

    # Save scalers and metadata
    try:
        with open(results_dir / 'scaler_x.pkl', 'wb') as f:
            pickle.dump(scaler_x, f)
        with open(results_dir / 'scaler_y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)
        with open(results_dir / 'meta.json', 'w') as f:
            import json
            json.dump({'target': target_col, 'seq_len': seq_len, 'features': feature_cols}, f, indent=2)
        # Save predictions
        pred_df = pd.DataFrame({
            'actual': y_test_orig,
            'predicted': y_pred_orig,
            'error': y_test_orig - y_pred_orig,
        })
        pred_df.to_csv(results_dir / 'predictions.csv', index=False)
    except Exception as exc:
        logger.warning(f'Failed to persist scalers/meta: {exc}')

    # Save metrics
    try:
        import json
        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass

    logger.info('Q5 Torch Transformer trained. RMSE=%.4f, R2=%.3f', metrics['rmse'], metrics['r2'])
    return metrics
