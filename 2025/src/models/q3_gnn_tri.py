from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import logging
import numpy as np
import pandas as pd

from utils.config import RESULTS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv
    TORCH_OK = True
except Exception:
    TORCH_OK = False


def _compute_segment_targets(trade_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute proxy targets for multi-task heads by segment.
    - trade_risk_target: normalized segment degree (country->segment weights)
    - supply_risk_target: normalized HHI of country shares within segment
    """
    # Aggregate c->s weights
    agg = trade_df.groupby(['partner_country', 'segment'])['chip_import_charges'].sum().reset_index()
    seg_total = agg.groupby('segment')['chip_import_charges'].sum()
    # Degree per segment
    seg_deg = agg.groupby('segment')['chip_import_charges'].sum()
    if len(seg_deg) == 0:
        return {}, {}
    seg_deg_norm = (seg_deg - seg_deg.min()) / (seg_deg.max() - seg_deg.min() + 1e-6)
    trade_risk_target = {seg: float(seg_deg_norm.get(seg, 0.0)) for seg in seg_deg_norm.index}

    # HHI per segment
    supply_risk_target: Dict[str, float] = {}
    for seg, total in seg_total.items():
        sub = agg[agg['segment'] == seg]
        if total <= 0:
            supply_risk_target[seg] = 0.0
            continue
        shares = (sub['chip_import_charges'] / total).values
        hhi = float((shares ** 2).sum())  # in [0,1]
        supply_risk_target[seg] = (hhi - 0.0) / (1.0 - 0.0 + 1e-6)  # normalize
    return trade_risk_target, supply_risk_target


def _build_tri_hetero(trade_df: pd.DataFrame, companies_per_pair: int = 2):
    countries = sorted(trade_df['partner_country'].dropna().unique().tolist())
    segments = sorted(trade_df['segment'].dropna().unique().tolist())
    c2i = {c: i for i, c in enumerate(countries)}
    s2i = {s: i for i, s in enumerate(segments)}

    data = HeteroData()
    # Simple initial features
    data['country'].x = torch.ones((len(countries), 4), dtype=torch.float32)
    data['segment'].x = torch.eye(len(segments), dtype=torch.float32)

    # Create synthetic company nodes: one per (country, segment, k)
    company_tuples: List[Tuple[str, str, int]] = []
    # Weights on country->segment
    agg = trade_df.groupby(['partner_country', 'segment'])['chip_import_charges'].sum().reset_index()
    for _, row in agg.iterrows():
        c = row['partner_country']
        s = row['segment']
        for k in range(companies_per_pair):
            company_tuples.append((c, s, k))
    if not company_tuples:
        raise ValueError('No edges to build tri-hetero graph')

    comp2i = {t: i for i, t in enumerate(company_tuples)}
    data['company'].x = torch.ones((len(company_tuples), 3), dtype=torch.float32)

    # country->segment edges (supplies)
    src_cs = [c2i[row['partner_country']] for _, row in agg.iterrows()]
    dst_cs = [s2i[row['segment']] for _, row in agg.iterrows()]
    w_cs = agg['chip_import_charges'].values.astype(np.float32)
    w_cs_norm = (w_cs - w_cs.min()) / (w_cs.max() - w_cs.min() + 1e-6)
    data[('country', 'supplies', 'segment')].edge_index = torch.tensor([src_cs, dst_cs], dtype=torch.long)
    data[('country', 'supplies', 'segment')].edge_weight = torch.tensor(w_cs_norm, dtype=torch.float32)

    # company->segment edges (makes): distribute evenly within each (c,s)
    src_cmp = []
    dst_cmp = []
    w_cmp = []
    for (c, s, k), idx in comp2i.items():
        src_cmp.append(idx)
        dst_cmp.append(s2i[s])
        # weight share as 1/companies_per_pair scaled by c->s normalized weight
        cs_weight = w_cs_norm[(agg['partner_country'] == c) & (agg['segment'] == s)]
        csw = float(cs_weight.iloc[0]) if len(cs_weight) else 0.0
        w_cmp.append(csw / companies_per_pair)
    data[('company', 'makes', 'segment')].edge_index = torch.tensor([src_cmp, dst_cmp], dtype=torch.long)
    data[('company', 'makes', 'segment')].edge_weight = torch.tensor(w_cmp, dtype=torch.float32)

    # country->company edges (hosts)
    src_host = []
    dst_host = []
    for (c, s, k), idx in comp2i.items():
        src_host.append(c2i[c])
        dst_host.append(idx)
    data[('country', 'hosts', 'company')].edge_index = torch.tensor([src_host, dst_host], dtype=torch.long)

    return data, c2i, s2i, comp2i


class TriHeteroGNN(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.convs = nn.ModuleList([
            HeteroConv({
                ('country', 'supplies', 'segment'): SAGEConv((-1, -1), hidden),
                ('company', 'makes', 'segment'): SAGEConv((-1, -1), hidden),
                ('country', 'hosts', 'company'): SAGEConv((-1, -1), hidden),
            }, aggr='sum'),
            HeteroConv({
                ('country', 'supplies', 'segment'): SAGEConv((-1, -1), hidden),
                ('company', 'makes', 'segment'): SAGEConv((-1, -1), hidden),
                ('country', 'hosts', 'company'): SAGEConv((-1, -1), hidden),
            }, aggr='sum'),
        ])
        # Multi-task heads on segment embeddings
        self.head_trade = nn.Linear(hidden, 1)
        self.head_supply = nn.Linear(hidden, 1)
        self.head_elast = nn.Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict):
        h = x_dict
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {k: F.relu(v) for k, v in h.items()}
        seg_h = h['segment']
        return {
            'segment_emb': seg_h,
            'trade_risk': self.head_trade(seg_h).squeeze(-1),
            'supply_risk': self.head_supply(seg_h).squeeze(-1),
            'elasticity': self.head_elast(seg_h).squeeze(-1),
        }


def _contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    # z1, z2: (N, D)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = torch.mm(z1, z2.t()) / tau  # (N,N)
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss = F.cross_entropy(sim, labels)
    return loss


def run_q3_tri_gnn(results_dir: Path, trade_df: pd.DataFrame) -> Dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    if not TORCH_OK:
        logger.warning('torch/torch_geometric not available; skipping tri-type GNN.')
        return {}

    torch.manual_seed(RANDOM_SEED)

    # Build data and targets
    data, c2i, s2i, comp2i = _build_tri_hetero(trade_df)
    trade_target, supply_target = _compute_segment_targets(trade_df)

    model = TriHeteroGNN(hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    # Prepare static targets tensor aligned by s2i order
    seg_list = sorted(s2i.items(), key=lambda x: x[1])
    y_trade = torch.tensor([trade_target.get(seg, 0.0) for seg, _ in seg_list], dtype=torch.float32)
    y_supply = torch.tensor([supply_target.get(seg, 0.0) for seg, _ in seg_list], dtype=torch.float32)

    # Elasticity proxy from YoY changes (if possible); else zeros
    try:
        tmp = trade_df.groupby(['year', 'segment'])['chip_import_charges'].sum().unstack().sort_index()
        yoy = tmp.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        elast_proxy = yoy.tail(3).mean().fillna(0.0)
        y_elast = torch.tensor([float(elast_proxy.get(seg, 0.0)) for seg, _ in seg_list], dtype=torch.float32)
    except Exception:
        y_elast = torch.zeros_like(y_trade)

    # Training loop
    for step in range(200):
        model.train()
        opt.zero_grad()
        out = model({'country': data['country'].x, 'segment': data['segment'].x, 'company': data['company'].x}, {
            ('country', 'supplies', 'segment'): data[('country', 'supplies', 'segment')].edge_index,
            ('company', 'makes', 'segment'): data[('company', 'makes', 'segment')].edge_index,
            ('country', 'hosts', 'company'): data[('country', 'hosts', 'company')].edge_index,
        })

        # Supervised losses
        loss_trade = F.mse_loss(out['trade_risk'], y_trade)
        loss_supply = F.mse_loss(out['supply_risk'], y_supply)
        loss_elast = F.mse_loss(out['elasticity'], y_elast)

        # Contrastive regularization: two edge-drop views
        def drop_edges(edge_index, drop=0.1):
            E = edge_index.shape[1]
            keep = int((1 - drop) * E)
            idx = torch.randperm(E)[:keep]
            return edge_index[:, idx]

        with torch.no_grad():
            ei1 = {k: drop_edges(v, 0.1) for k, v in {
                ('country', 'supplies', 'segment'): data[('country', 'supplies', 'segment')].edge_index,
                ('company', 'makes', 'segment'): data[('company', 'makes', 'segment')].edge_index,
                ('country', 'hosts', 'company'): data[('country', 'hosts', 'company')].edge_index,
            }.items()}
            ei2 = {k: drop_edges(v, 0.1) for k, v in ei1.items()}
        out1 = model({'country': data['country'].x, 'segment': data['segment'].x, 'company': data['company'].x}, ei1)
        out2 = model({'country': data['country'].x, 'segment': data['segment'].x, 'company': data['company'].x}, ei2)
        loss_con = _contrastive_loss(out1['segment_emb'], out2['segment_emb'])

        # Total loss with weights
        loss = 1.0 * loss_trade + 0.7 * loss_supply + 0.5 * loss_elast + 0.1 * loss_con
        loss.backward()
        opt.step()

    # Inference
    model.eval()
    with torch.no_grad():
        out = model({'country': data['country'].x, 'segment': data['segment'].x, 'company': data['company'].x}, {
            ('country', 'supplies', 'segment'): data[('country', 'supplies', 'segment')].edge_index,
            ('company', 'makes', 'segment'): data[('company', 'makes', 'segment')].edge_index,
            ('country', 'hosts', 'company'): data[('country', 'hosts', 'company')].edge_index,
        })

    # Save outputs
    seg_names = [seg for seg, _ in seg_list]
    df_scores = pd.DataFrame({
        'segment': seg_names,
        'trade_risk': out['trade_risk'].cpu().numpy().tolist(),
        'supply_risk': out['supply_risk'].cpu().numpy().tolist(),
        'elasticity': out['elasticity'].cpu().numpy().tolist(),
    })
    df_scores.to_csv(results_dir / 'tri_gnn_segment_scores.csv', index=False)

    torch.save({'state_dict': model.state_dict()}, results_dir / 'tri_gnn.pt')

    summary = {
        'method': 'tri_hetero_gnn_pyg',
        'segments': seg_names,
        'heads': ['trade_risk', 'supply_risk', 'elasticity'],
        'contrastive': True,
    }
    with open(results_dir / 'tri_gnn_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
