from abc import ABC
import argparse
import json
import os
import pickle
from typing import Any, Dict, List
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
from matplotlib import pyplot as plt
import omegaconf
import pandas as pd
import yaml
import mbrl.constants
import mbrl.types
import mbrl.util.common as common
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac_pranz24 import SAC
from mbrl.third_party.pytorch_sac import VideoRecorder
import mbrl.util.mujoco
from omegaconf import OmegaConf
import mbrl.util.logger
from citylearn.data import DataSet
from citylearn.citylearn import CityLearnEnv
import seaborn as sns
from mbrl.util.kpi_utils import evaluate_citylearn_challenge, get_kpis
from mbrl.rewards.CityLearnReward import SolarPenaltyAndComfortReward
from citylearn.wrappers import StableBaselines3Wrapper
from citylearn.agents.rbc import OptimizedRBC
from mbrl.util.plot_utils import compare_kpis, plot_energy, plot_temperature

CHALLENGE_WEIGHTS_PHASE_CUSTOM = {
    'comfort': 0.3,
    'emissions': 0.4,
    'grid_control': 0.3,
    'resilience': 0.0
}
PATH = '/home/lorenzobonanni/Desktop/Model-Based-Reinforcement-Learning-for-energy-management-in-smart-buildings/src/agents/model_based/outputs/2026-01-15/16-15-34/macura/sac.pth'

class ComfortRBC(OptimizedRBC):
    """
    Rule-based Control designed to overwrite controls scheduled by :py:class:`citylearn.agents.rbc.OptimizedRBC` 
    in order to tackle temperature discomfort.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment to perform control on.
    band: float
        Comfort band to try to satisfy. 

    TODO
    ---------- 
    Understand how to manage storages and devices with respect to them
    """
    def __init__(self, env: CityLearnEnv, band: float=None, **kwargs):        
        # Init OptimizedRBC
        super().__init__(env, **kwargs)

        # Sanity check
        self._check(env)

        # Comfort band (+/-) to satisfy
        self.comfort_band = band if band is not None else env.buildings[0].comfort_band[0] 

    def predict(self, observations: List[List[float]], deterministic: bool=None) -> List[List[float]]:        
        # Predict actions based on hour scheduling
        scheduled_acions = super().predict(observations, deterministic)

        actions = []
        for i, o in enumerate(observations):
            action = scheduled_acions[i]

            # Available spaces
            available_obs = self.observation_names[i]
            available_act = self.action_names[i]

            # Temperatures
            if 'indoor_dry_bulb_temperature' in available_obs:
                indoor_temp = o[available_obs.index('indoor_dry_bulb_temperature')]
            else:
                indoor_temp = None

            if 'outdoor_dry_bulb_temperature' in available_obs:
                outdoor_temp = o[available_obs.index('outdoor_dry_bulb_temperature')]
            else:
                outdoor_temp = None

            if 'indoor_dry_bulb_temperature_cooling_set_point' in available_obs:
                cooling_setpoint = o[available_obs.index('indoor_dry_bulb_temperature_cooling_set_point')]
            else:
                cooling_setpoint = None

            if 'indoor_dry_bulb_temperature_cooling_delta' in available_obs:
                cooling_delta = o[available_obs.index('indoor_dry_bulb_temperature_cooling_delta')]
            else:
                cooling_delta = None

            if 'indoor_dry_bulb_temperature_heating_set_point' in available_obs:
                heating_setpoint = o[available_obs.index('indoor_dry_bulb_temperature_heating_set_point')]
            else:
                heating_setpoint = None

            if 'indoor_dry_bulb_temperature_heating_delta' in available_obs:
                heating_delta = o[available_obs.index('indoor_dry_bulb_temperature_heating_delta')]
            else:
                heating_delta = None

            # Stoarges SoC
            if 'electrical_storage_soc' in available_obs:
                electrical_soc = o[available_obs.index('electrical_storage_soc')]
            else:
                electrical_soc = -1

            if 'cooling_storage_soc' in available_obs:
                cooling_soc = o[available_obs.index('cooling_storage_soc')]
            else:
                cooling_soc = -1

            if 'heating_storage_soc' in available_obs:
                heating_soc = o[available_obs.index('heating_storage_soc')]
            else:
                heating_soc = -1

            # Manage cooling
            if 'cooling_device' in available_act:
                # Action indexes
                device_idx = available_act.index('cooling_device')
                if 'electircal_storage' in available_act:
                    ess_idx = available_act.index('electrical_storage')
                else:
                    ess_idx = None

                # Temperature difference
                hot_delta = cooling_delta if cooling_delta is not None else indoor_temp - cooling_setpoint
                if hot_delta > 0:
                    if hot_delta > self.comfort_band: # Too hot -> supply the cooling device                        
                        action[device_idx] = 0.8
                        if electrical_soc > 0.1 and ess_idx is not None:
                            action[ess_idx] =  min(action[ess_idx], -electrical_soc/2)
                    else:
                        action[device_idx] = 0.2 # Hot within the band
                        if electrical_soc > 0.1 and ess_idx is not None:
                            action[ess_idx] =  min(action[ess_idx], -electrical_soc/3)
                else:
                    if indoor_temp is not None and outdoor_temp is not None:
                        temp_delta = outdoor_temp - indoor_temp # Outdoor temperature affects indoor temperature                       
                        action[device_idx] = 0.3 if temp_delta > 0 else 0.0

                    else:
                        action[device_idx] = 0.0
                        if ess_idx is not None: 
                            action[ess_idx] = action[ess_idx]/2 if action[ess_idx] < 0 else action[ess_idx]

            # Manage heating
            if 'heating_device' in available_act:
                # Action indexes
                device_idx = available_act.index('heating_device')
                if 'electrical_storage' in available_act:
                    ess_idx = available_act.index('electrical_storage')
                else:
                    ess_idx = None

                # Temperature difference
                cold_delta = heating_delta if heating_delta is not None else indoor_temp - heating_setpoint
                if cold_delta < 0:
                    if cold_delta < -self.comfort_band:
                        action[device_idx] = 0.8 # Too cold -> supply the heating device
                        if electrical_soc > 0.1 and ess_idx is not None:
                            action[ess_idx] =  min(action[ess_idx], -electrical_soc/2)
                    else:
                        action[device_idx] = 0.2 # Cold within the band
                        if electrical_soc > 0.1 and ess_idx is not None:
                            action[ess_idx] =  min(action[ess_idx], -electrical_soc/3)
                else:
                    if indoor_temp is not None and outdoor_temp is not None:
                        temp_delta = outdoor_temp - indoor_temp # Outdoor temperature affects indoor temperature
                        action[device_idx] = 0.3 if temp_delta < 0 else 0.0
                    else:
                        action[device_idx] = 0.0
                        if ess_idx is not None: 
                            action[ess_idx] = action[ess_idx]/2 if action[ess_idx] < 0 else action[ess_idx]

            actions.append(action)

        # Return overwritten actions
        self.actions = actions
        return actions
    
    def _check(self, env: CityLearnEnv):
        if 'indoor_dry_bulb_temperature' in env.observation_names[0]:
            if 'indoor_dry_bulb_temperature_cooling_set_point' not in env.observation_names[0] \
                and 'indoor_dry_bulb_temperature_heating_set_point' not in env.observation_names[0] \
                and 'indoor_dry_bulb_temperature_cooling_delta' not in env.observation_names[0] \
                and 'indoor_dry_bulb_temperature_heating_delta' not in env.observation_names[0]:
                raise RuntimeError(
                    '`indoor_dry_bulb_temperature` is available, but no `indoor_dry_bulb_temperature_*_set_point` ' +
                    'or  `indoor_dry_bulb_temperature_*_delta` is available.'
                )
        else:
            if 'indoor_dry_bulb_temperature_cooling_delta' not in env.observation_names[0] \
                and 'indoor_dry_bulb_temperature_heating_delta' not in env.observation_names[0]:
                raise RuntimeError('No `indoor_dry_bulb_temperature_*_delta` is available.')


class CityLearnSchema:
    def __init__(self, schema: Dict[str, Any] | None=None):
        self._schema: Dict[str, Any] | None = schema

    @property
    def schema(self) -> Dict[str, Any]:
        return self._schema
    
    @schema.setter
    def schema(self, new_schema: Dict[str, Any]):
        self._schema = new_schema

    def load(self, dataset: str, custom: bool=False):
        assert self._schema is None, 'Schema has already been loaded.'

        # Get the schema of the dataset
        self._schema = DataSet().get_schema(dataset)

        if custom:
            print("="*40)
            print("CityLearnOmnisafe - Dataset Customization")
            print("="*40)
            print(f"Dataset: {dataset}")

            # User's building selection
            _ = self._select_items(key='buildings')
            # User's observation selection
            selected_obs = self._select_items(key='observations')
            # User's action selection
            selected_act = self._select_items(key='actions')

            # Sanity check
            self._check(selected_obs, selected_act)
        else:
            # Include Building_1 by default
            self.set_active(key='buildings', items=['Building_1'])
            self._schema['observations']['cooling_electricity_consumption']['active'] = True
            self._schema['observations']['dhw_electricity_consumption']['active'] = True

    def save(self, dir: str, prefix: str='base'):
        with open(f'{dir}/{prefix}_schema.json', 'w') as f:
            json.dump(self._schema, f, indent=4)

    def set(self, key: str, value: Dict[str, Any]):
        self._schema[key] = value

    def set_active(self, key: str, items: List[str]):
        assert self._schema is not None, 'Schema has not been loaded, yet.'
        assert key in ['buildings', 'observations', 'actions'], f'Unknown schema key {key}.'

        # Filter CityLearn items
        flag_key = 'include' if key == 'buildings' else 'active'
        for it in self._schema[key].keys():
            self._schema[key][it][flag_key] = (it in items)

    def train_test_split(self, frac: float, mode: str):
        assert mode in ['train', 'test'], f'Unknown mode {mode}. Must be either `train` or `test`.'
        assert 0 < frac <= 1, f'Invalid fraction {frac}. Must be in (0,1).'

        # Copy base schema
        train_schema, test_schema = self._schema.copy(), self._schema.copy()

        # Total simulation days
        time_steps = self._schema['simulation_end_time_step'] + 1
        total_days = time_steps // 24

        # Train/test split index
        train_days = int(total_days * frac)
        split_idx = train_days * 24

        # Modify train/test schemas
        train_schema['simulation_end_time_step'] = split_idx - 1
        if frac < 1:
            test_schema['simulation_start_time_step'] = split_idx

        return train_schema, test_schema

    def _select_items(self, key: str):
        assert key in ['buildings', 'observations', 'actions'], f'Unknown schema key {key}.'
    
        # Available items
        if key == 'buildings':
            pool = list(self._schema[key].keys())
        else:
            pool = [item for item in self._schema[key].keys() if self._schema[key][item]['active']]

        print(f"Available {key}:")
        for idx, item in enumerate(pool):
            print(f"- {idx+1}. {item}")

        # Item selection
        user_input = input(f"\nSelect {item} by entering their numbers separated by commas (e.g., 1,3,5): ")
        selected_indices = [int(i.strip()) - 1 for i in user_input.split(',') if i.strip().isdigit() and 0 < int(i.strip()) <= len(pool)]
        selected_items = [pool[i] for i in selected_indices]

        print(f"Selected items: {selected_items}\n\n")

        # Modify schema according to user's selection
        self.set_active(key=key, items=selected_items)

        return selected_items
    
    def _check(self, observations: List[str], actions: List[str]):
        print('Checking observations...')
        if 'indoor_dry_bulb_temperature' in observations:
            if 'indoor_dry_bulb_temperature_cooling_set_point':
                # Remove "redundant" observations
                observations.remove('indoor_dry_bulb_temperature_cooling_set_point')

                # Activate temperature delta
                observations.append('indoor_dry_bulb_temperature_cooling_delta')
                self.set_active(key='observations', items=observations)
                print(
                    '[CHECK] Both `indoor_dry_bulb_temperature` and `indoor_dry_bulb_temperature_cooling_set_point` are active.' + 
                    ' `indoor_dry_bulb_temperature_cooling_delta` has been activated.'
                )

            if 'indoor_dry_bulb_temperature_heating_set_point' in observations:
                # Remove "reduntant" observations
                observations.remove('indoor_dry_bulb_temperature_heating_set_point')

                # Activate temperature delta
                observations.append('indoor_dry_bulb_temperature_heating_delta')
                self.set_active(key='observations', items=observations)
                print(
                    '[CHECK] Both `indoor_dry_bulb_temperature` and `indoor_dry_bulb_temperature_heating_set_point` are active.' + 
                    ' `indoor_dry_bulb_temperature_heating_delta` has been activated.'
                )


def evaluate(agent: object, env: object, seed: int=None):

    # elif agent_type == 'advanced_rbc':
    #     agent = AdvancedRBC(env)
    # elif agent_type == 'custom_rbc':
    #     # agent = CustomRBC(env)
    #     agent = ComfortRBC(env)
    # else:
    #     raise RuntimeError(f'Unknown agent type {agent_type}. Must be either `rbc` or `rl`.')
    
    # Episodic return
    results = {}
    ep_reward = 0.0

    # Step through the environment
    obs, _ = env.reset(seed=seed)
    while not env.terminated:
        if isinstance(agent, SACAgent):
            action = agent.act(obs)
        else:
            action = agent.predict([obs])
            action = action[0]
        obs, reward, _, _, _ = env.step(action)
        ep_reward += reward

    # Get KPIs
    kpis = get_kpis(env=env)

    # Console log
    print(
        f"{'*'*30}\n CONTROL RESULTS (RL{f' | seed={seed}' if seed is not None else ''})" +
        f'\n- Reward: {ep_reward}'
    )

    for kpi, value in kpis.items():
        print(f'- {kpi}: {value:.2f}')

    print(f"{'*'*30}")

    # Populate results dict
    results['kpis'] = kpis
    results['env_h'] = {
        'time_steps': env.time_steps,
        'temperature': {
            'indoor_temperature': env.buildings[0].indoor_dry_bulb_temperature,
            'indoor_temperature_set_point': env.buildings[0].indoor_dry_bulb_temperature_cooling_set_point,
            'outdoor_temperature': env.buildings[0].weather.outdoor_dry_bulb_temperature ,
            'comfort_band': env.buildings[0].comfort_band
        },
        'battery': {
            'soc': env.buildings[0].electrical_storage.soc[:-1],
            'discharge': env.buildings[0].electrical_storage.energy_balance[:-1],
            'consumption': env.buildings[0].electrical_storage.electricity_consumption[:-1]
        },
        'dhw': {
            'soc': env.buildings[0].dhw_storage.soc[:-1],
            'demand': env.buildings[0].dhw_demand[:-1],
            'consumption': env.buildings[0].dhw_electricity_consumption[:-1],
            'energy_from_dhw_storage': env.buildings[0].energy_from_dhw_storage[:-1],
            'energy_from_dhw_device': env.buildings[0].energy_from_dhw_device[:-1]
        },
        'cooling_device': {
            'consumption': env.buildings[0].cooling_device.electricity_consumption[:-1]
        },
        'net_electricity_consumption': env.buildings[0].net_electricity_consumption[:-1],
        'solar_generation': env.buildings[0].solar_generation[:-1],
        'non_shiftable_load': env.buildings[0].non_shiftable_load[:-1],
        'electricity_pricing': env.buildings[0].pricing.electricity_pricing[:-1],
    }

    return results


@hydra.main(config_path="conf", config_name="launcher_macura")
def main(launcher_cfg: omegaconf.DictConfig):
    """Top-level Hydra entrypoint (the one you call via `python -m`)."""
    global agent

    train_cfg_name = launcher_cfg.train_cfg
    test_cfg_name = launcher_cfg.test_cfg

    # Get schema from CityLearn dataset
    schema_obj = CityLearnSchema()
    data = 'citylearn_challenge_2023_phase_1'
    schema_obj.load(dataset=data)

    # Modify schema for testing on a different building
    schema_obj.set_active(key='buildings', items=[f'Building_2'])


    GlobalHydra.instance().clear()
    with initialize(config_path="conf"):
        cfg = compose(config_name=train_cfg_name)
            # Create CityLearn environment
        env = CityLearnEnv(
            schema=schema_obj.schema, 
            central_agent=True,
        )
        env = StableBaselines3Wrapper(env)
        reward_fn = SolarPenaltyAndComfortReward(env.schema)
        env.reward_function = reward_fn
        mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
        sac_impl = SAC(
            cfg.algorithm.agent.num_inputs,
            env.action_space,
            cfg.algorithm.agent.args,
        )
        agent = SACAgent(sac_impl)
        agent.sac_agent.load_checkpoint(ckpt_path=PATH)
        
        # RL agent evaluation
        res_rl = evaluate(agent, env, seed=0)
        plot_temperature(res_rl, suffix='rl')
        plot_energy(res_rl, cfg.algorithm.name)
        scores_rl = evaluate_citylearn_challenge(
            env,
            weights=CHALLENGE_WEIGHTS_PHASE_CUSTOM
        )

        # Comfort RBC evaluation
        res_rbc = evaluate(ComfortRBC(env.unwrapped), env, seed=0)
        plot_temperature(res_rbc, suffix='comfort_rbc')
        plot_energy(res_rbc, "comfort_rbc")
        scores_rbc = evaluate_citylearn_challenge(
            env,
            weights=CHALLENGE_WEIGHTS_PHASE_CUSTOM
        )
        
        compare_kpis(res_rbc, res_rl, algo_names=['Comfort RBC', cfg.algorithm.name])
        workdir = os.getcwd()
        with open(os.path.join(workdir, f"{cfg.algorithm.name}_rl_results.pkl"), 'wb') as f:
            pickle.dump(res_rl, f)
        
        scores_rl_df = pd.DataFrame.from_dict(scores_rl, orient='index')
        scores_rl_df.to_csv(os.path.join(workdir, f"{cfg.algorithm.name}_rl_scores.csv"))
        scores_rbc_df = pd.DataFrame.from_dict(scores_rbc, orient='index')
        scores_rbc_df.to_csv(os.path.join(workdir, f"comfort_rbc_scores.csv"))

if __name__ == "__main__":
    main()