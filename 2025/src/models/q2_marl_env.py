import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from utils.config import RESULTS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)


class AutoMarketEnv:
    """A minimal MARL environment for the US–Japan auto policy game.

    State (vector): [us_tariff, jp_relocation, import_penetration, us_production_norm, employment_norm]
    Actions:
        - US (agent 0): tariff level in [0, 0.30]
        - Japan (agent 1): relocation intensity in [0, 1]

    Dynamics are simplified and consistent with the econometric scenario logic used in q2.
    """

    def __init__(self,
                 baseline_japan_sales: float = 2_000_000.0,
                 baseline_us_production: float = 10_000_000.0,
                 baseline_employment: float = 950_000.0):
        self.baseline_japan_sales = float(baseline_japan_sales)
        self.baseline_us_production = float(baseline_us_production)
        self.baseline_employment = float(baseline_employment)
        self.state = None
        self.last_components: Dict[str, Dict[str, float]] = {}
        self.reset()

    def reset(self) -> np.ndarray:
        # Start from no tariff and no relocation
        us_tariff = 0.0
        jp_relocation = 0.0
        import_penetration, us_prod, emp = self._compute_economy(us_tariff, jp_relocation)
        self.state = np.array([us_tariff, jp_relocation, import_penetration, us_prod, emp], dtype=np.float32)
        return self.state.copy()

    def step(self, action_us: float, action_jp: float) -> Tuple[np.ndarray, Tuple[float, float]]:
        # Clamp actions to valid ranges
        action_us = float(np.clip(action_us, 0.0, 0.30))
        action_jp = float(np.clip(action_jp, 0.0, 1.0))

        import_penetration, us_prod, emp = self._compute_economy(action_us, action_jp)
        self.state = np.array([action_us, action_jp, import_penetration, us_prod, emp], dtype=np.float32)

        # Rewards with decomposition
        us_reward, jp_reward = self._compute_rewards(action_us, action_jp, import_penetration, us_prod, emp)
        return self.state.copy(), (us_reward, jp_reward)

    # --- Internal helpers ---
    def _compute_economy(self, us_tariff: float, jp_relocation: float) -> Tuple[float, float, float]:
        # Shares follow scenario heuristics:
        # - Direct imports from Japan shrink with relocation
        # - US production share grows with relocation
        # - Mexico share absorbs the remainder
        japan_direct_share = max(0.0, 0.30 * (1.0 - 0.8 * jp_relocation))
        us_produced_share = np.clip(0.20 + 0.40 * jp_relocation, 0.0, 0.90)
        mexico_share = np.clip(1.0 - (japan_direct_share + us_produced_share), 0.0, 1.0)

        # Effective imports exposed to tariff: only Japan direct
        japan_direct_imports = self.baseline_japan_sales * japan_direct_share
        total_imports = japan_direct_imports + self.baseline_japan_sales * mexico_share

        # Import penetration and induced change in US production
        total_supply = self.baseline_us_production + total_imports
        import_penetration = float(total_imports / max(total_supply, 1e-6))

        # Assume 1%↑ penetration -> 0.5%↓ US production (centered at 25%)
        production_impact = -0.5 * (import_penetration - 0.25) * self.baseline_us_production
        new_us_prod = max(0.0, self.baseline_us_production + production_impact)
        new_emp = self.baseline_employment * (new_us_prod / max(self.baseline_us_production, 1e-6))

        # Normalize production and employment to [0,1] scale for rewards
        us_production_norm = float(new_us_prod / max(self.baseline_us_production, 1e-6))
        employment_norm = float(new_emp / max(self.baseline_employment, 1e-6))
        return import_penetration, us_production_norm, employment_norm

    def _compute_rewards(self, us_tariff: float, jp_relocation: float,
                         import_penetration: float, us_prod_norm: float, emp_norm: float) -> Tuple[float, float]:
        # US tariff revenue proxy: tariff * Japan direct imports (normalized by baseline)
        japan_direct_share = max(0.0, 0.30 * (1.0 - 0.8 * jp_relocation))
        japan_direct_imports = self.baseline_japan_sales * japan_direct_share
        revenue_proxy = us_tariff * (japan_direct_imports / max(self.baseline_japan_sales, 1e-6))

        # Consumer surplus proxy: price increase on direct imports harms consumers
        consumer_surplus = -0.5 * us_tariff * japan_direct_share  # negative is welfare loss

        # Domestic firm profit proxy: align with domestic production gain
        delta_prod = max(0.0, us_prod_norm - 1.0)
        firm_profit = 0.8 * delta_prod

        # Employment gain
        employment_gain = (emp_norm - 1.0)

        # Aggregate US reward (weights can be tuned)
        us_reward = 80.0 * employment_gain + 15.0 * revenue_proxy + 30.0 * firm_profit + 40.0 * consumer_surplus

        # Japan reward: sales retention - relocation cost
        sales_retention = 1.0 - 0.2 * us_tariff - 0.15 * jp_relocation  # heuristics
        relocation_cost = 0.30 * jp_relocation  # cost share
        jp_profit = sales_retention - relocation_cost
        jp_reward = jp_profit * 100.0

        # Store decomposition for external logging
        self.last_components = {
            'us': {
                'fiscal_revenue': float(revenue_proxy),
                'employment_gain': float(employment_gain),
                'consumer_surplus': float(consumer_surplus),
                'domestic_firm_profit': float(firm_profit),
            },
            'jp': {
                'sales_retention': float(sales_retention),
                'relocation_cost': float(relocation_cost),
                'firm_profit': float(jp_profit),
            }
        }

        return float(us_reward), float(jp_reward)


class SelfPlayTrainer:
    """Iterative best-response over discrete grids (self-play approximation)."""

    def __init__(self, env: AutoMarketEnv,
                 us_tariff_grid: np.ndarray = None,
                 jp_relocation_grid: np.ndarray = None,
                 n_iters: int = 20):
        self.env = env
        self.us_grid = us_tariff_grid if us_tariff_grid is not None else np.linspace(0.0, 0.30, 7)
        self.jp_grid = jp_relocation_grid if jp_relocation_grid is not None else np.linspace(0.0, 1.0, 6)
        self.n_iters = int(n_iters)
        self.history: List[Dict[str, Any]] = []

    def _evaluate(self, us_a: float, jp_a: float) -> Tuple[float, float]:
        _ = self.env.reset()
        _, (r_us, r_jp) = self.env.step(us_a, jp_a)
        return r_us, r_jp

    def run(self) -> Dict[str, Any]:
        np.random.seed(RANDOM_SEED)
        us_action = float(self.us_grid[len(self.us_grid) // 2])
        jp_action = float(self.jp_grid[len(self.jp_grid) // 2])

        for it in range(self.n_iters):
            # Best response for US given Japan action
            us_rewards = []
            for a in self.us_grid:
                r_us, _ = self._evaluate(a, jp_action)
                us_rewards.append((a, r_us))
            us_action = float(max(us_rewards, key=lambda x: x[1])[0])

            # Best response for Japan given US action
            jp_rewards = []
            for b in self.jp_grid:
                _, r_jp = self._evaluate(us_action, b)
                jp_rewards.append((b, r_jp))
            jp_action = float(max(jp_rewards, key=lambda x: x[1])[0])

            r_us, r_jp = self._evaluate(us_action, jp_action)
            self.history.append({
                'iter': it + 1,
                'us_action': us_action,
                'jp_action': jp_action,
                'us_reward': r_us,
                'jp_reward': r_jp,
            })

        result = {
            'final_us_tariff': us_action,
            'final_jp_relocation': jp_action,
            'final_rewards': {'us': self.history[-1]['us_reward'], 'jp': self.history[-1]['jp_reward']},
            'convergence_path': self.history,
        }
        return result


def run_marl_env_training(results_dir: Path = RESULTS_DIR / 'q2' / 'marl') -> Dict[str, Any]:
    env = AutoMarketEnv()
    trainer = SelfPlayTrainer(env)
    out = trainer.run()

    results_dir.mkdir(parents=True, exist_ok=True)
    # Save CSV path
    hist_df = pd.DataFrame(trainer.history)
    hist_df.to_csv(results_dir / 'selfplay_history.csv', index=False)

    # Save JSON summary
    import json
    with open(results_dir / 'selfplay_summary.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    logger.info("MARL self-play training completed. Results saved to %s", results_dir)
    return out
