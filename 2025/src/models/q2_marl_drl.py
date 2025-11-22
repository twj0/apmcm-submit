import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from utils.config import RESULTS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    try:
        from .q2_marl_env import AutoMarketEnv  # type: ignore
    except Exception:
        from q2_marl_env import AutoMarketEnv  # type: ignore
except Exception:
    AutoMarketEnv = None  # type: ignore


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, out_act: str | None = None):
        super().__init__()
        act = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), act,
            nn.Linear(hidden, hidden), act,
            nn.Linear(hidden, out_dim)
        )
        self.out_act = out_act

    def forward(self, x):
        y = self.net(x)
        if self.out_act == 'tanh':
            return torch.tanh(y)
        return y


class MultiAgentReplayBuffer:
    """Replay buffer storing joint transitions for two agents.

    Stores (s, [a_us, a_jp], [r_us, r_jp], s'). Centralized critic will
    consume joint actions and joint state, while each agent uses its own
    scalar reward for SAC updates.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int, n_agents: int, device):
        self.capacity = int(capacity)
        self.ptr = 0
        self.full = False
        self.device = device
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.n_agents = int(n_agents)
        self.s = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.a = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.r = torch.zeros((capacity, n_agents), dtype=torch.float32, device=device)
        self.ns = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)

    def add(self, s, a, r_vec, ns):
        n = self.ptr
        self.s[n] = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        self.a[n] = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        self.r[n] = torch.as_tensor(r_vec, dtype=torch.float32, device=self.device)
        self.ns[n] = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        max_idx = self.capacity if self.full else self.ptr
        if max_idx == 0:
            raise RuntimeError("Buffer is empty, cannot sample")
        idx = np.random.randint(0, max_idx, size=batch_size)
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx]


class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy for continuous actions in [low, high]."""

    def __init__(self, state_dim: int, action_low: float, action_high: float,
                 hidden: int = 128, log_std_min: float = -5.0, log_std_max: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden, 1)
        self.log_std_layer = nn.Linear(hidden, 1)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        # scaling to env bounds
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.action_scale = 0.5 * (self.action_high - self.action_low)
        self.action_bias = 0.5 * (self.action_high + self.action_low)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(s)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action with reparameterization; return action, log_prob, mu, log_std."""
        mu, log_std = self(s)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre_tanh = mu + eps * std
        a_tanh = torch.tanh(pre_tanh)
        # scale to [low, high]
        action = a_tanh * self.action_scale + self.action_bias

        # Log prob with tanh correction
        # base Gaussian log_prob
        log_prob = -0.5 * (((pre_tanh - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # Tanh squashing correction
        log_prob -= torch.log(torch.clamp(1 - a_tanh.pow(2), min=1e-6)).sum(dim=-1, keepdim=True)
        return action, log_prob, mu, log_std


class CentralizedCritic(nn.Module):
    """Q(s, a_us, a_jp) for a single agent (its own reward)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.q_net = MLP(state_dim + action_dim, 1, hidden)

    def forward(self, s: torch.Tensor, a_joint: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a_joint], dim=-1)
        return self.q_net(x)


class MultiAgentSAC:
    """Two-agent SAC with centralized critics and decentralized actors.

    - Two actors: US (tariff in [0, 0.30]) and JP (relocation in [0, 1]).
    - For each agent k ∈ {US, JP} we keep twin Q networks Q1_k, Q2_k
      that take joint (s, a_us, a_jp) as input.
    - Training uses standard SAC losses with per-agent entropy terms.
    """

    def __init__(self, state_dim: int,
                 us_action_low: float, us_action_high: float,
                 jp_action_low: float, jp_action_high: float,
                 device, gamma: float = 0.99, tau: float = 0.005,
                 alpha_us: float = 0.1, alpha_jp: float = 0.1):
        self.device = device
        self.state_dim = int(state_dim)
        self.action_dim = 2  # [a_us, a_jp]
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.alpha_us = float(alpha_us)
        self.alpha_jp = float(alpha_jp)

        # Actors (decentralized execution)
        self.actor_us = GaussianPolicy(state_dim, us_action_low, us_action_high).to(device)
        self.actor_jp = GaussianPolicy(state_dim, jp_action_low, jp_action_high).to(device)

        # Centralized critics (twin Q for stability) — one pair per agent
        self.q1_us = CentralizedCritic(state_dim, self.action_dim).to(device)
        self.q2_us = CentralizedCritic(state_dim, self.action_dim).to(device)
        self.q1_us_tgt = CentralizedCritic(state_dim, self.action_dim).to(device)
        self.q2_us_tgt = CentralizedCritic(state_dim, self.action_dim).to(device)

        self.q1_jp = CentralizedCritic(state_dim, self.action_dim).to(device)
        self.q2_jp = CentralizedCritic(state_dim, self.action_dim).to(device)
        self.q1_jp_tgt = CentralizedCritic(state_dim, self.action_dim).to(device)
        self.q2_jp_tgt = CentralizedCritic(state_dim, self.action_dim).to(device)

        # Initialize targets
        self.q1_us_tgt.load_state_dict(self.q1_us.state_dict())
        self.q2_us_tgt.load_state_dict(self.q2_us.state_dict())
        self.q1_jp_tgt.load_state_dict(self.q1_jp.state_dict())
        self.q2_jp_tgt.load_state_dict(self.q2_jp.state_dict())

        # Optimizers
        self.opt_actor_us = torch.optim.Adam(self.actor_us.parameters(), lr=3e-4)
        self.opt_actor_jp = torch.optim.Adam(self.actor_jp.parameters(), lr=3e-4)
        self.opt_q_us = torch.optim.Adam(
            list(self.q1_us.parameters()) + list(self.q2_us.parameters()), lr=3e-4
        )
        self.opt_q_jp = torch.optim.Adam(
            list(self.q1_jp.parameters()) + list(self.q2_jp.parameters()), lr=3e-4
        )

    def act(self, s: np.ndarray, eval_mode: bool = False) -> Tuple[float, float]:
        """Sample joint actions for US and JP given current state (numpy)."""
        self.actor_us.eval()
        self.actor_jp.eval()
        with torch.no_grad():
            s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            if eval_mode:
                mu_us, _ = self.actor_us(s_t)
                mu_jp, _ = self.actor_jp(s_t)
                a_us = torch.tanh(mu_us) * self.actor_us.action_scale + self.actor_us.action_bias
                a_jp = torch.tanh(mu_jp) * self.actor_jp.action_scale + self.actor_jp.action_bias
            else:
                a_us, _, _, _ = self.actor_us.sample(s_t)
                a_jp, _, _, _ = self.actor_jp.sample(s_t)
            return float(a_us.squeeze(0).item()), float(a_jp.squeeze(0).item())

    def _soft_update(self, net, target_net):
        with torch.no_grad():
            for p, pt in zip(net.parameters(), target_net.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def train_step(self, buf: MultiAgentReplayBuffer, batch_size: int = 128):
        if (not buf.full and buf.ptr < batch_size) or (buf.full and buf.capacity < batch_size):
            return

        s, a, r, ns = buf.sample(batch_size)

        # --- Critic updates (centralized) ---
        with torch.no_grad():
            # Next actions from current policies
            a_us_next, logp_us_next, _, _ = self.actor_us.sample(ns)
            a_jp_next, logp_jp_next, _, _ = self.actor_jp.sample(ns)
            a_next_joint = torch.cat([a_us_next, a_jp_next], dim=-1)

            # US targets
            q1_us_tgt_next = self.q1_us_tgt(ns, a_next_joint)
            q2_us_tgt_next = self.q2_us_tgt(ns, a_next_joint)
            q_us_tgt = torch.min(q1_us_tgt_next, q2_us_tgt_next)
            y_us = r[:, 0:1] + self.gamma * (q_us_tgt - self.alpha_us * logp_us_next)

            # JP targets
            q1_jp_tgt_next = self.q1_jp_tgt(ns, a_next_joint)
            q2_jp_tgt_next = self.q2_jp_tgt(ns, a_next_joint)
            q_jp_tgt = torch.min(q1_jp_tgt_next, q2_jp_tgt_next)
            y_jp = r[:, 1:2] + self.gamma * (q_jp_tgt - self.alpha_jp * logp_jp_next)

        # Current Q estimates
        q1_us = self.q1_us(s, a)
        q2_us = self.q2_us(s, a)
        q1_jp = self.q1_jp(s, a)
        q2_jp = self.q2_jp(s, a)

        loss_q_us = F.mse_loss(q1_us, y_us) + F.mse_loss(q2_us, y_us)
        loss_q_jp = F.mse_loss(q1_jp, y_jp) + F.mse_loss(q2_jp, y_jp)

        self.opt_q_us.zero_grad()
        loss_q_us.backward()
        self.opt_q_us.step()

        self.opt_q_jp.zero_grad()
        loss_q_jp.backward()
        self.opt_q_jp.step()

        # --- Actor updates (decentralized policies, centralized critic) ---
        # US actor
        a_us, logp_us, _, _ = self.actor_us.sample(s)
        with torch.no_grad():
            a_jp_eval, _, _, _ = self.actor_jp.sample(s)
        a_joint_us = torch.cat([a_us, a_jp_eval], dim=-1)
        q1_us_pi = self.q1_us(s, a_joint_us)
        actor_us_loss = (self.alpha_us * logp_us - q1_us_pi).mean()

        self.opt_actor_us.zero_grad()
        actor_us_loss.backward()
        self.opt_actor_us.step()

        # JP actor
        a_jp, logp_jp, _, _ = self.actor_jp.sample(s)
        with torch.no_grad():
            a_us_eval, _, _, _ = self.actor_us.sample(s)
        a_joint_jp = torch.cat([a_us_eval, a_jp], dim=-1)
        q1_jp_pi = self.q1_jp(s, a_joint_jp)
        actor_jp_loss = (self.alpha_jp * logp_jp - q1_jp_pi).mean()

        self.opt_actor_jp.zero_grad()
        actor_jp_loss.backward()
        self.opt_actor_jp.step()

        # --- Target network updates ---
        self._soft_update(self.q1_us, self.q1_us_tgt)
        self._soft_update(self.q2_us, self.q2_us_tgt)
        self._soft_update(self.q1_jp, self.q1_jp_tgt)
        self._soft_update(self.q2_jp, self.q2_jp_tgt)


def run_marl_drl_training(results_dir: Path = RESULTS_DIR / 'q2' / 'marl',
                          episodes: int = 200,
                          steps_per_ep: int = 8) -> Dict[str, Any]:
    """Train two SAC agents (US tariff, JP relocation) in AutoMarketEnv.

    Uses centralized critics over joint state and actions, with
    decentralized stochastic policies for each agent (CTDE paradigm).
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    if not TORCH_OK or AutoMarketEnv is None:
        logger.warning('PyTorch or environment not available; skipping DRL training.')
        return {}

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    env = AutoMarketEnv()
    state_dim = 5  # [us_tariff, jp_relocation, import_penetration, us_prod_norm, emp_norm]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sac = MultiAgentSAC(
        state_dim=state_dim,
        us_action_low=0.0,
        us_action_high=0.30,
        jp_action_low=0.0,
        jp_action_high=1.0,
        device=dev,
    )

    buf = MultiAgentReplayBuffer(50_000, state_dim, action_dim=2, n_agents=2, device=dev)

    history = []

    s = env.reset()
    for ep in range(episodes):
        s = env.reset()
        ep_ret_us = 0.0
        ep_ret_jp = 0.0
        # accumulate reward components
        us_comp_sum = {'fiscal_revenue': 0.0, 'employment_gain': 0.0, 'consumer_surplus': 0.0, 'domestic_firm_profit': 0.0}
        jp_comp_sum = {'sales_retention': 0.0, 'relocation_cost': 0.0, 'firm_profit': 0.0}
        for t in range(steps_per_ep):
            # Each step both agents act based on current state (decentralized policies)
            a_us, a_jp = sac.act(s, eval_mode=False)
            ns, (r_us, r_jp) = env.step(a_us, a_jp)

            # Store joint transition and train SAC (centralized critic)
            buf.add(
                s,
                np.array([a_us, a_jp], dtype=np.float32),
                np.array([r_us, r_jp], dtype=np.float32),
                ns,
            )

            sac.train_step(buf, batch_size=128)

            s = ns
            ep_ret_us += r_us
            ep_ret_jp += r_jp
            # sum components
            if getattr(env, 'last_components', None):
                uc = env.last_components.get('us', {})
                jc = env.last_components.get('jp', {})
                for k in us_comp_sum:
                    us_comp_sum[k] += float(uc.get(k, 0.0))
                for k in jp_comp_sum:
                    jp_comp_sum[k] += float(jc.get(k, 0.0))

        # averages per step
        denom = float(max(1, steps_per_ep))
        row = {'episode': ep + 1, 'us_return': ep_ret_us, 'jp_return': ep_ret_jp}
        row.update({f"us_{k}": v / denom for k, v in us_comp_sum.items()})
        row.update({f"jp_{k}": v / denom for k, v in jp_comp_sum.items()})
        history.append(row)

    # Save actor artifacts (policies)
    try:
        torch.save({'actor': sac.actor_us.state_dict()}, results_dir / 'sac_us_actor.pt')
        torch.save({'actor': sac.actor_jp.state_dict()}, results_dir / 'sac_jp_actor.pt')
    except Exception as exc:
        logger.warning(f'Failed to save SAC actor weights: {exc}')

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(results_dir / 'drl_training_curve.csv', index=False)

    summary = {
        'episodes': episodes,
        'steps_per_episode': steps_per_ep,
        'final_us_return': history[-1]['us_return'] if history else None,
        'final_jp_return': history[-1]['jp_return'] if history else None,
    }
    import json
    with open(results_dir / 'drl_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info('MARL DRL (SAC) training completed.')
    return summary
