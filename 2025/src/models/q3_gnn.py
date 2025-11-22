from pathlib import Path
from typing import Dict, Any, Tuple
import json
import logging
import numpy as np
import pandas as pd

from utils.config import RESULTS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

# Optional imports (graceful fallback)
try:
    import torch
    from torch import nn
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv
    TORCH_OK = True
except Exception:
    TORCH_OK = False


def _build_hetero_data(trade_df: pd.DataFrame) -> Tuple[Any, Dict[str, int], Dict[str, int]]:
    countries = sorted(trade_df['partner_country'].dropna().unique().tolist())
    segments = sorted(trade_df['segment'].dropna().unique().tolist())
    c2i = {c: i for i, c in enumerate(countries)}
    s2i = {s: i for i, s in enumerate(segments)}

    data = HeteroData()
    data['country'].x = torch.ones((len(countries), 4), dtype=torch.float32)
    data['segment'].x = torch.eye(len(segments), dtype=torch.float32)

    # Aggregate across years
    agg = trade_df.groupby(['partner_country', 'segment'])['chip_import_charges'].sum().reset_index()
    src = [c2i[c] for c in agg['partner_country'] if c in c2i]
    dst = [s2i[s] for s in agg['segment'] if s in s2i]
    w = agg['chip_import_charges'].values.astype(np.float32)
    if len(src) == 0:
        raise ValueError('No edges to build hetero graph')

    data[('country', 'supplies', 'segment')].edge_index = torch.tensor([src, dst], dtype=torch.long)
    # Normalize weights
    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-6)
    data[('country', 'supplies', 'segment')].edge_weight = torch.tensor(w_norm, dtype=torch.float32)
    return data, c2i, s2i


class RiskGNN(nn.Module):
    def __init__(self, in_country: int, in_segment: int, hidden: int = 16):
        super().__init__()
        self.convs = nn.ModuleList([
            HeteroConv({('country', 'supplies', 'segment'): SAGEConv((-1, -1), hidden)}, aggr='sum'),
            HeteroConv({('country', 'supplies', 'segment'): SAGEConv((-1, -1), hidden)}, aggr='sum'),
        ])
        self.lin_seg = nn.Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        seg_score = self.lin_seg(x_dict['segment']).squeeze(-1)
        return x_dict, seg_score


def run_q3_full_gnn(results_dir: Path, trade_df: pd.DataFrame) -> Dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    if not TORCH_OK:
        logger.warning('torch/torch_geometric not available; skipping full GNN.')
        return {}

    torch.manual_seed(RANDOM_SEED)
    try:
        data, c2i, s2i = _build_hetero_data(trade_df)
    except Exception as exc:
        logger.warning(f'Failed to build hetero graph: {exc}')
        return {}

    model = RiskGNN(data['country'].x.shape[1], data['segment'].x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Simple edge-weight reconstruction loss: predict segment score to match mean of incident weights
    seg_deg = torch.zeros((data['segment'].x.shape[0],), dtype=torch.float32)
    idx = data[('country', 'supplies', 'segment')].edge_index
    w = data[('country', 'supplies', 'segment')].edge_weight
    for j, weight in zip(idx[1].tolist(), w.tolist()):
        seg_deg[j] += weight
    target = seg_deg / (seg_deg.max() + 1e-6)

    # Increase training epochs to 500 for better convergence
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    for epoch in range(500):  # Increased from 100 to 500
        model.train()
        opt.zero_grad()
        _, pred = model({'country': data['country'].x, 'segment': data['segment'].x},
                        {('country', 'supplies', 'segment'): data[('country', 'supplies', 'segment')].edge_index})
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        opt.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}, best loss: {best_loss:.6f}")
                break
        
        # Log progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            logger.info(f"GNN training epoch {epoch+1}/500, loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        emb_dict, seg_score = model({'country': data['country'].x, 'segment': data['segment'].x},
                                    {('country', 'supplies', 'segment'): data[('country', 'supplies', 'segment')].edge_index})

    # Save outputs
    seg_list = sorted(s2i.items(), key=lambda x: x[1])
    seg_scores = pd.DataFrame({'segment': [k for k, _ in seg_list],
                               'risk_score': seg_score.cpu().numpy().tolist()})
    seg_scores.to_csv(results_dir / 'gnn_segment_scores.csv', index=False)

    torch.save({
        'state_dict': model.state_dict(),
    }, results_dir / 'risk_gnn.pt')

    summary = {
        'method': 'hetero_gnn_pyg',
        'segments': list(s2i.keys()),
    }
    with open(results_dir / 'gnn_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
