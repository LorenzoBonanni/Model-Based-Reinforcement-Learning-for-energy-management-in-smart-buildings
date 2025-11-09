import os
import sys
import hydra
from hydra import initialize, compose
import omegaconf
import numpy as np
import pandas as pd
import torch
import wandb

# Allow importing from mbrl root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mbrl.util.kpi_utils import evaluate_citylearn_challenge, plot_simulation_summary
import mbrl.algorithms.macura as macura
import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.m2ac as m2ac
import mbrl.util.env as env_util
import mbrl.algorithms.sac as sac
import mbrl.util.common
from hydra.core.global_hydra import GlobalHydra
from mbrl.third_party.pytorch_sac import VideoRecorder

global agent


def run_experiment(train_cfg_name):
    """Loads a training config and runs the correct algorithm."""

    # TODO: compute TRAIN KPIs and return them
    GlobalHydra.instance().clear()
    with initialize(config_path="conf"):
        cfg = compose(config_name=train_cfg_name)

        print(f"Using algorithm: {cfg.algorithm.name}")
        env, term_fn, reward_fn = env_util.EnvHandler.make_env(cfg, test_env=False)

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        wandb.init(
            project="ModelBased",
            config=omegaconf.OmegaConf.to_container(cfg),
        )

        # Select and run algorithm
        if cfg.algorithm.name == "mbpo":
            test_env, *_ = env_util.EnvHandler.make_env(cfg, test_env=True)
            return mbpo.train(env, test_env, term_fn, cfg)

        elif cfg.algorithm.name == "m2ac":
            test_env, *_ = env_util.EnvHandler.make_env(cfg, test_env=True)
            return m2ac.train(env, test_env, term_fn, cfg)

        elif cfg.algorithm.name == "macura":
            test_env, *_ = env_util.EnvHandler.make_env(cfg, test_env=True)
            test_env2, *_ = env_util.EnvHandler.make_env(cfg, test_env=True)
            return macura.train(env, test_env, test_env2, term_fn, cfg)
        
        elif cfg.algorithm.name == "sac":
            test_env, *_ = env_util.EnvHandler.make_env(cfg, test_env=True)
            return sac.train(env, test_env, term_fn, cfg)

        else:
            raise ValueError(f"Unknown algorithm: {cfg.algorithm.name}")


def test_experiment(test_cfg_name):
    """Loads a test config and evaluates the global agent."""
    global agent
    print("----------------------------------------")
    print("Testing the best agent found during training...")
    print("----------------------------------------")

    GlobalHydra.instance().clear()
    with initialize(config_path="conf"):
        cfg = compose(config_name=test_cfg_name)

        test_env, *_ = env_util.EnvHandler.make_env(cfg, test_env=True)
        avg_reward, results = mbrl.util.common.final_evaluate(
            test_env,
            agent,
            cfg.algorithm.num_eval_episodes,
            cfg.algorithm.name
        )

        workdir = os.getcwd()
        infos = pd.DataFrame(results)
        mean_infos = infos.mean()
        std_infos = infos.std()
        agg_infos = pd.concat([mean_infos, std_infos], axis=1, keys=['mean', 'std'])
        agg_infos.to_csv(os.path.join(workdir, f"{cfg.algorithm.name}_test_kpis.csv"))
        infos.to_csv(os.path.join(workdir, f"{cfg.algorithm.name}_test_kpis_episodes.csv"))

@hydra.main(config_path="conf", config_name="launcher_macura")
def main(launcher_cfg: omegaconf.DictConfig):
    """Top-level Hydra entrypoint (the one you call via `python -m`)."""
    global agent

    train_cfg_name = launcher_cfg.train_cfg
    test_cfg_name = launcher_cfg.test_cfg

    print(f"Launcher config:\n  train_cfg={train_cfg_name}\n  test_cfg={test_cfg_name}")

    # Run training
    best_score, best_agent = run_experiment(train_cfg_name)

    # Store globally (as in your original code)
    agent = best_agent

    # Run test
    test_experiment(test_cfg_name)


if __name__ == "__main__":
    main()