import os
import sys
import numpy as np
import pandas as pd
from citylearn.citylearn import CityLearnEnv
from agents.model_based.mbrl.util.CityLearnWrappers import CityLearnKPIWrapper
from agents.model_based.mbrl.util.plot_utils import make_plots
from rewards.CityLearnReward import SolarPenaltyAndComfortReward
from agents.checa.agent import Checa as Agent

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
    idx_battery_action = np.where(np.array(eval_env.unwrapped.buildings[0].active_actions) == "electrical_storage")[0][0]
    idx_cooling_action = np.where(np.array(eval_env.unwrapped.buildings[0].active_actions) == "cooling_device")[0][0]
    idx_dhw_action = np.where(np.array(eval_env.unwrapped.buildings[0].active_actions) == "dhw_storage")[0][0]

    # Agent
    model = Agent(eval_env)
    eval_env = CityLearnKPIWrapper(eval_env)


    # Number of the chosen building and number of episode trainig
    # num_episodes = 5
    num_episodes = 5

    initial_workdir = os.getcwd()
    now = datetime.now()
    date_part = now.strftime("%Y-%m-%d")
    time_part = now.strftime("%H-%M-%S")
    target_dir = os.path.join(initial_workdir, "agents", "checa", "outputs", date_part, time_part)
    os.makedirs(target_dir, exist_ok=True)
    os.chdir(target_dir)
    workdir = os.getcwd()

    os.chdir(initial_workdir)
    # Evaluate
    infos = []
    for episode in range(num_episodes):
        observations, _ = eval_env.reset(seed=episode)
        terminated = truncated = False
        battery_actions = [[0.0] for _ in range(len(eval_env.unwrapped.buildings))]
        cooling_actions = [[0.0] for _ in range(len(eval_env.unwrapped.buildings))]
        dhw_actions = [[0.0] for _ in range(len(eval_env.unwrapped.buildings))]
        while not terminated and not truncated:
            actions = model.predict([observations])
            actions = actions[0]
            observations, reward, terminated, truncated, info = eval_env.step(actions)
            for i in range(len(eval_env.unwrapped.buildings)):
                battery_actions[i].append(actions[3*i + idx_battery_action])
                cooling_actions[i].append(actions[3*i + idx_cooling_action])
                dhw_actions[i].append(actions[3*i + idx_dhw_action])
            if info:
                infos.append(info)
        print(f"Episode {episode + 1} finished.")
        make_plots(eval_env, cooling_actions=cooling_actions, battery_actions=battery_actions, dhw_actions=dhw_actions, output_dir=workdir, episode=episode)
        make_plots(eval_env, cooling_actions=cooling_actions, battery_actions=battery_actions, dhw_actions=dhw_actions, output_dir=workdir, episode=episode, limit=24*10)  # First 10 days
        building_kpi, district_kpi = plot_simulation_summary({'Checa': eval_env}, workdir, algorithm_name)

        score = evaluate_citylearn_challenge(
            eval_env,
            phase_1_weights
        )

        pd.DataFrame(score).to_csv(os.path.join(workdir, f"{algorithm_name}_test_score.csv"))
        building_kpi.to_csv(os.path.join(workdir, f"{algorithm_name}_building_kpis.csv"))
        district_kpi.to_csv(os.path.join(workdir, f"{algorithm_name}_district_kpis.csv"))



    infos = pd.DataFrame(infos)
    mean_infos = infos.mean()
    std_infos = infos.std()
    agg_infos = pd.concat([mean_infos, std_infos], axis=1, keys=['mean', 'std'])
    agg_infos.to_csv(os.path.join(workdir, f"{algorithm_name}_test_kpis.csv"))
    infos.to_csv(os.path.join(workdir, f"{algorithm_name}_test_kpis_episodes.csv"))
    

if __name__ == "__main__":
    test_experiment()