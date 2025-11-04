import os
import sys
import pandas as pd
from citylearn.citylearn import CityLearnEnv
from rewards.CityLearnReward import SolarPenaltyAndComfortReward
from agents.checa.agent import Agent

# Allow importing from mbrl root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.model_based.mbrl.util.kpi_utils import evaluate_citylearn_challenge, plot_simulation_summary
from datetime import datetime

def test_experiment():
    """Loads a test config and evaluates the global agent."""
    print("----------------------------------------")
    print("Testing the best agent found during training...")
    print("----------------------------------------")
    algorithm_name = "checa"

    env_name = "citylearn_challenge_2023_phase_2_online_evaluation_3"
    eval_env = CityLearnEnv(env_name, central_agent=True)
    rf = SolarPenaltyAndComfortReward(eval_env.schema)
    eval_env.reward_function = rf
    eval_env.random_seed=0

    # Agent
    model = Agent(eval_env)

    # Number of the chosen building and number of episode trainig
    # num_episodes = 5
    num_episodes = 5

    # Evaluate
    infos = []
    for episode in range(num_episodes):
        observations, _ = eval_env.reset(seed=episode)
        terminated = truncated = False
        while not terminated and not truncated:
            actions = model.predict(observations, deterministic=True)
            observations, reward, terminated, truncated, info = eval_env.step(actions)
            if info:
                infos.append(info)

    workdir = os.getcwd()
    now = datetime.now()
    date_part = now.strftime("%Y-%m-%d")
    time_part = now.strftime("%H-%M-%S")
    target_dir = os.path.join(workdir, "agents", "checa", "outputs", date_part, time_part)
    os.makedirs(target_dir, exist_ok=True)
    os.chdir(target_dir)
    workdir = os.getcwd()

    print("WORKDIR", workdir)
    infos = pd.DataFrame(infos)
    mean_infos = infos.mean()
    std_infos = infos.std()
    agg_infos = pd.concat([mean_infos, std_infos], axis=1, keys=['mean', 'std'])
    agg_infos.to_csv(os.path.join(workdir, f"{algorithm_name}_test_kpis.csv"))
    infos.to_csv(os.path.join(workdir, f"{algorithm_name}_test_kpis_episodes.csv"))
    building_kpi, district_kpi = plot_simulation_summary({'Checa': eval_env}, workdir, algorithm_name)

    phase_1_weights = {
        'comfort': 0.3,
        'emissions': 0.1,
        'grid_control': 0.6,
        'resilience': 0.0
    }
    phase_2_weights = {
        'comfort': 0.3,
        'emissions': 0.1,
        'grid_control': 0.3,
        'resilience': 0.3
    }
    custom_weights = {
        'comfort': 0.3,
        'emissions': 0.4,
        'grid_control': 0.3,
        'resilience': 0.0
    }
    score = evaluate_citylearn_challenge(
        eval_env,
        phase_1_weights
    )

    pd.DataFrame(score).to_csv(os.path.join(workdir, f"{algorithm_name}_test_score.csv"))
    building_kpi.to_csv(os.path.join(workdir, f"{algorithm_name}_building_kpis.csv"))
    district_kpi.to_csv(os.path.join(workdir, f"{algorithm_name}_district_kpis.csv"))


if __name__ == "__main__":
    test_experiment()