import gymnasium
import numpy as np
import pandas as pd
import os
import datetime
import time
from sklearn.model_selection import train_test_split
# from stable_baselines3 import PPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from typing import Optional, Set, Dict, Tuple, Any, Iterable, List
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.spaces import MultiDiscrete, Box
from sb3_contrib.ppo_mask import MaskablePPO
import simbench as sb
import copy
from gymnasium import spaces
import pandapower as pp
import warnings
import json
from pandapower.auxiliary import pandapowerNet
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import time
from tqdm import tqdm


warnings.simplefilter(action='ignore', category=FutureWarning)

class ENV_MaskableRHV(gymnasium.Env):
    """
    Reinforcement Learning Environment for Reconfigurable High Voltage Network with Enhanced Observation.

    This environment allows for reconfiguration actions over a SimBench high-voltage network and computes
    rewards based on voltage stability, line loading, and power flow feasibility.

    Parameters
    ----------
    simbench_code : str
        SimBench case code (e.g., "1-HV-mixed--0-sw").
    case_study : str
        Identifier for the case study scenario (e.g., 'bc').
    is_train : bool
        If True, loads training profiles; otherwise, loads testing profiles.
    is_normalize : bool
        Whether to normalize observations.
    max_step : int
        Maximum number of steps per episode.
    allowed_lines : int
        Maximum number of allowed connected lines.
    convergence_penalty : float
        Penalty for power flow convergence failure.
    line_disconnect_penalty : float
        Penalty for each disconnected line.
    nan_vm_pu_penalty : float
        Penalty when voltage magnitude becomes NaN.
    reward_type : str
        Reward strategy type (e.g., "Original", "PenaltyOnly").
    rho_min : float
        Minimum allowed line loading rate (0–1).
    action_type : str
        Strategy for defining the action space (e.g., 'HotSpots2').
    exp_code : str
        experiment code tag for tracking/logging/runs.
    penalty_scalar : float
        Scalar multiplier for general penalties.
    bonus_constant : float
        Constant reward bonus for good configurations.
    hot_spots : Set[int]
        Set of bus indices where busbar reassignment is allowed.
    """

    def __init__(self,
                 simbench_code: str = "RHVModV1",
                 case_study: str = 'bc',
                 is_train: bool = True,
                 is_normalize: bool = False,
                 max_step: int = 96,
                 allowed_lines: int = 200,
                 convergence_penalty: float = -200,
                 line_disconnect_penalty: float = -200,
                 nan_vm_pu_penalty: float = -200,
                 reward_type: str = "Original",
                 rho_min: float = 0.45,
                 action_type: str = 'EncMask_CBandAllLinesFltrd',
                 exp_code: str = None,
                 include_features: List[str] = ["time"],
                 penalty_scalar: float = -10,
                 bonus_constant: float = 10,
                 shuffle: bool = False,
                 hot_spots: Set[int] = set()) -> None:
        super().__init__()

        # --- Configuration Parameters ---
        self.simbench_code = simbench_code
        self.case_study = case_study
        self.is_train = is_train
        self.is_normalize = is_normalize
        self.max_step = max_step
        self.reward_type = reward_type
        self.action_type = action_type
        self.bonus_constant = bonus_constant
        self.rho_min = rho_min
        self.exp_code = exp_code
        self.penalty_scalar = penalty_scalar
        self.hot_spots = hot_spots
        self.shuffle = shuffle
        self.include_features = include_features

        # Generate a warning that the shuffling is used
        if self.shuffle:
          print("[NOTE:] Shuffling of data is used in the environment.")


        # --- Penalties ---
        self.allowed_lines = allowed_lines
        self.convergence_penalty = convergence_penalty
        self.line_disconnect_penalty = line_disconnect_penalty
        self.nan_vm_pu_penalty = nan_vm_pu_penalty

        # --- State Variables ---
        self.net = self.load_simbench_net()
        self.initial_net = None
        self.relative_index = None
        self.time_step = -1
        self.observation = None
        self.truncated = False
        self.terminated = False
        self.info: Dict[str, Any] = {}
        self.count = 0

        # --- Data Placeholders ---
        self.profiles = None
        self.gen_data = None
        self.load_data = None
        self.sgen_data = None

        # --- Error Counters ---
        self.line_disconnect_count = 0
        self.convergence_error_count = 0
        self.nan_vm_pu_count = 0

        # --- Time Control ---
        self.override_timestep = None
        self.test_data_length = None
        self.train_data_length = None

        # --- Action Type Specific ---
        if self.action_type in ["HotSpots2", "EncMask_CBandAllLinesFltrd"]:
            self.line_switch_dict = self.get_line_switch_dict(self.net)
            self.switch_df = self.get_line_switch_dataframe(self.net)
            self.switch_pair_map=self.build_switch_pair_map(self.switch_df)
            # # --- Initial Penalty Feedback ---
            # invalid_count = self.check_validity(self.net, self.switch_df)
            # invalid_penalty = self.penalty_scalar * invalid_count
            # print(f"Invalid penalty: {invalid_penalty}")

        # --- Load and Setup ---
        self.initial_net = self.set_study_case(case_study, self.is_train, load_all=True)
        self.action_space, self.observation_space = self.create_act_obs_space()
        _ = self.reset()

        # --- RL Hyperparameters ---
        self.gamma = 0.99
        self.rho_max = 1.0



        # Sleep for debugging or startup sync
        time.sleep(5)

    def action_masks(self):
        """
        Returns a boolean mask for a MultiDiscrete action space
        """

        if self.action_type == "EncMask_CBandAllLinesFltrd":
            # Get dimensions
            num_cbs = len(self.cb_excluding_ehv)
            num_lines = len(self.line_ids)

            # Create mask list for MultiDiscrete action space
            mask_list = []

            # For circuit breakers: allow both actions [0, 1]
            for cb_idx in range(num_cbs):
                mask_list.append([True, True])  # Both actions 0 and 1 are allowed

            # For line switches: mask encoding 3 (which represents (1,1))
            for line_idx in range(num_lines):
                mask_list.append([True if i in [0, 5, 6, 9, 10] else False for i in range(16)])  # Block action 3

            # Flatten the mask to match MultiDiscrete format
            flattened_mask = []
            for action_mask in mask_list:
                flattened_mask.extend(action_mask)

            flattened_mask = np.array(flattened_mask, dtype=bool)

            return flattened_mask
        else:
            return np.array([True] * self.action_space.nvec.sum())



    def load_simbench_net(self) -> pandapowerNet:
        """
        Loads the SimBench network with BESS units.
        First tries to load from a JSON file named after the simbench_code.
        If the file is not found or invalid, attempts to load using the SimBench API.
        Raises a clear error if both methods fail.
        """

        errors = []

        # Try loading from JSON
        try:
            return pp.from_json(f"{self.simbench_code}.json")
        except Exception as e_json:
            errors.append(f"JSON load failed: {str(e_json)}")

        # Try loading using SimBench code
        try:
            return sb.get_simbench_net(self.simbench_code)
        except Exception as e_sb:
            errors.append(f"SimBench code load failed: {str(e_sb)}")

        # If both fail
        raise ValueError(
            f"Failed to load SimBench network for code '{self.simbench_code}'. "
            f"Details:\n" + "\n".join(errors)
        )

    def set_study_case(self, case_study: str, is_train: bool, load_all: bool = True) -> Optional[pandapowerNet]:
        """
        Configures the study case for the environment by applying absolute values to the network
        and loading generation and load profiles. It also handles data normalization and splits the
        profiles into training and testing sets.

        Parameters
        ----------
        case_study : str
            Identifier for the case study.
        is_train : bool
            If True, training data will be loaded. If False, testing data is used.
        load_all : bool, optional
            If True, loads and splits all the profile data. Defaults to True.

        Returns
        -------
        Optional[pandapowerNet]
            Updated pandapower network if load_all is True; otherwise None.
        """
        if not load_all:
            return None

        print("Initializing study case...")

        self.case_study = case_study

        # Apply absolute values to the network based on case study
        loadcases = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=False)
        net = self.apply_absolute_values(self.net, loadcases, self.case_study)

        # Load time-series profile data
        self.profiles = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=True)
        load_data_raw = self.profiles[('load', 'p_mw')]
        sgen_data_raw = self.profiles[('sgen', 'p_mw')]
        storage_data_raw = self.profiles[('storage', 'p_mw')]


        # Fill NaNs with 0
        self.load_data_normalized = load_data_raw.fillna(0)
        self.sgen_data_normalized = sgen_data_raw.fillna(0)
        self.storage_data_normalized = storage_data_raw.fillna(0)

        # Split into training and testing sets (80% train, 20% test)
        load_train, load_test = train_test_split(self.load_data_normalized, test_size=0.2, shuffle=self.shuffle,random_state=42)
        sgen_train, sgen_test = train_test_split(self.sgen_data_normalized, test_size=0.2, shuffle=self.shuffle,random_state=42)
        storage_train, storage_test = train_test_split(self.storage_data_normalized, test_size=0.2, shuffle=self.shuffle,random_state=42)

        print(load_train.head())
        # Store data lengths
        self.train_data_length = sgen_train.shape[0]
        self.test_data_length = sgen_test.shape[0]

        if is_train:
            # Build environment metadata for reproducibility and logging
            self.env_meta = {
                "is_train": self.is_train,
                "is_normalize": self.is_normalize,
                "max_step": self.max_step,
                "case_study": self.case_study,
                "simbench_code": self.simbench_code,
                "allowed_lines": self.allowed_lines,
                "convergence_penalty": self.convergence_penalty,
                "line_disconnect_penalty": self.line_disconnect_penalty,
                "nan_vm_pu_penalty": self.nan_vm_pu_penalty,
                "rho_min": self.rho_min,
                "Shuffle": self.shuffle,
                "include_features": self.include_features,
                "train_data_length": self.train_data_length,
                "test_data_length": self.test_data_length,
                "load_data_shape": self.load_data_normalized.shape,
                "sgen_data_shape": self.sgen_data_normalized.shape,
                "storage_train_data_shape": storage_train.shape,
                "storage_test_data_shape": storage_test.shape,
                "load_train_data_shape": load_train.shape,
                "sgen_train_data_shape": sgen_train.shape,
                "load_test_data_shape": load_test.shape,
                "sgen_test_data_shape": sgen_test.shape,
                "exp_code": self.exp_code,
                "action_type": self.action_type,
                "reward_type": self.reward_type,
                "penalty_scalar": self.penalty_scalar,
                "bonus_constant": self.bonus_constant,
                "total_CBs": self.net.switch[(self.net.switch['et'] == 'b') & (self.net.switch['type'] == 'CB')].shape[0],
                "total_switches": self.net.switch.shape[0],
                "num_cbs_excluding_EHVCBs": self.get_cbs_excluding_ehv().shape[0],
                "hot_spots": list(self.hot_spots)
            }

            # Save metadata to file
            with open("env_meta.json", "w") as file:
                json.dump(self.env_meta, file, indent=4)
            print("Environment metadata saved to env_meta.json.")
            print(self.env_meta)

            # Assign training data
            self.load_data = load_train
            self.sgen_data = sgen_train
            self.storage_data = storage_train


        else:
            # Assign testing data
            self.load_data = load_test
            self.sgen_data = sgen_test
            self.storage_data = storage_test

        return net

    def get_cbs_excluding_ehv(self) -> pd.DataFrame:
        """
        Retrieves all circuit breakers (CBs) that are not connected to extra high voltage (EHV) buses,
        i.e., buses with nominal voltage of 220 kV or 380 kV.

        Returns
        -------
        pd.DataFrame
            DataFrame of CB switches excluding those connected to EHV buses.
        """
        # Filter switches that are circuit breakers connected to buses (et='b')
        cb_switches = self.net.switch[
            (self.net.switch['et'] == 'b') & (self.net.switch['type'] == 'CB')
        ]

        def is_non_ehv_bus(bus_id: int) -> bool:
            """Checks if a bus has nominal voltage not in EHV range (220 or 380 kV)."""
            vn_kv = self.net.bus.loc[bus_id, 'vn_kv']
            return vn_kv not in {220, 380}

        # Apply filter to keep only CBs not connected to any EHV buses
        valid_cbs = cb_switches[
            cb_switches.apply(
                lambda row: is_non_ehv_bus(row['bus']) and is_non_ehv_bus(row['element']),
                axis=1
            )
        ]

        return valid_cbs

    def get_hot_spots_switch_ids(self, hot_spots_buses: Iterable[int], net: pandapowerNet) -> Set[int]:
        """
        Retrieves the set of switch IDs that are connected to the specified hot spot buses.

        Parameters
        ----------
        hot_spots_buses : Iterable[int]
            A collection of bus indices considered as hot spots.
        net : pandapowerNet
            The pandapower network containing the buses and switches.

        Returns
        -------
        Set[int]
            A set of switch indices that are connected to the hot spot buses.
        """
        hot_spots_switch_ids = set()

        for bus_id in hot_spots_buses:
            # Get all busbar-type switches (type 'b') connected to the bus
            bus_switches = pp.get_connected_switches(net, bus_id, consider=('b'), status="all")
            hot_spots_switch_ids.update(bus_switches)

        return hot_spots_switch_ids

    def get_line_switch_dict(self, net: pandapowerNet) -> Dict[int, Dict[str, Any]]:
        """
        Creates a dictionary mapping each line to its associated switches on the from-bus and to-bus sides.

        Switches are collected using all types ('b', 'l', 's', 't') and include all statuses.

        Parameters
        ----------
        net : pandapowerNet
            The pandapower network from which to extract line and switch relationships.

        Returns
        -------
        Dict[int, Dict[str, Any]]
            A dictionary with line IDs as keys and dictionaries of 'from_bus' and 'to_bus' switch lists as values.
            Lines with only one switch on either side are excluded.
        """
        line_switch_dict: Dict[int, Dict[str, Any]] = {}

        for line_id, line_row in net.line.iterrows():
            from_bus = line_row['from_bus']
            to_bus = line_row['to_bus']

            # Get all switches connected to each bus (regardless of switch type or status)
            switches_from = pp.get_connected_switches(net, from_bus, consider=('b', 'l', 's', 't'), status='all')
            switches_to = pp.get_connected_switches(net, to_bus, consider=('b', 'l', 's', 't'), status='all')

            line_switch_dict[line_id] = {
                'from_bus': switches_from,
                'to_bus': switches_to
            }

        # Remove lines where either side has only a single switch (assumed not relevant)
        filtered_dict = {
            line_id: switches
            for line_id, switches in line_switch_dict.items()
            if len(switches['from_bus']) > 1 and len(switches['to_bus']) > 1
        }

        return filtered_dict

    def get_paired_switch(self, df: pd.DataFrame, switch_id: int) -> Optional[int]:
        """
        Finds the paired switch for a given switch ID from the provided DataFrame.

        The function checks if the given switch ID appears in one of the four columns:
        'SW_A_from', 'SW_B_from', 'SW_A_to', or 'SW_B_to', and returns its counterpart.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing switch pair information with expected columns.
        switch_id : int
            The ID of the switch whose pair is to be found.

        Returns
        -------
        Optional[int]
            The paired switch ID if found; otherwise, None.
        """
        for _, row in df.iterrows():
            if row['SW_A_from'] == switch_id:
                return row['SW_B_from']
            if row['SW_B_from'] == switch_id:
                return row['SW_A_from']
            if row['SW_A_to'] == switch_id:
                return row['SW_B_to']
            if row['SW_B_to'] == switch_id:
                return row['SW_A_to']

        return None


    def get_line_switch_dataframe(self, net: pandapowerNet) -> pd.DataFrame:
        """
        Constructs a DataFrame mapping each line in the network to up to two switches on both its
        from-bus and to-bus sides.

        Only lines with exactly two switches on each side will have both switch IDs populated;
        otherwise, `None` is used.

        Parameters
        ----------
        net : pandapowerNet
            The pandapower network containing line and switch data.

        Returns
        -------
        pd.DataFrame
            A DataFrame with each row representing a line and its associated switch IDs:
            - line_id
            - SW_A_from, SW_B_from
            - SW_A_to, SW_B_to
        """
        line_data_with_switches = []

        for line_id, line_row in net.line.iterrows():
            from_bus = line_row['from_bus']
            to_bus = line_row['to_bus']

            # Get all switches connected to both from and to buses
            switches_from = list(pp.get_connected_switches(net, from_bus, consider=('b', 'l', 's', 't'), status='all'))
            switches_to = list(pp.get_connected_switches(net, to_bus, consider=('b', 'l', 's', 't'), status='all'))

            # Assign switches only if exactly 2 are found
            sw_a_from, sw_b_from = (switches_from if len(switches_from) == 2 else (None, None))
            sw_a_to, sw_b_to = (switches_to if len(switches_to) == 2 else (None, None))

            line_data_with_switches.append({
                'line_id': line_id,
                'SW_A_from': sw_a_from,
                'SW_B_from': sw_b_from,
                'SW_A_to': sw_a_to,
                'SW_B_to': sw_b_to
            })

        return pd.DataFrame(line_data_with_switches)



    def create_act_obs_space(self):
        """
        Creates and returns the action and observation spaces for the environment,
        based on the specified action type.

        Returns:
            Tuple[gym.Space, gym.Space]: A tuple containing the action space and observation space.
        """
        print("_______ Creating action and observation space for action type: {self.action_type} _______")

        cb_switches = self.net.switch[(self.net.switch['et'] == 'b') & (self.net.switch['type'] == 'CB')]
        action_space = None

        if self.action_type == "NodeSplitting":
            self.cb_switches = cb_switches
            action_space = spaces.MultiDiscrete([2] * cb_switches.shape[0])

        elif self.action_type == "NodeSplittingExEHVCBs":
            self.cb_switches = cb_switches
            self.cb_excluding_ehv = self.get_cbs_excluding_ehv()
            action_space = spaces.MultiDiscrete([2] * self.cb_excluding_ehv.shape[0])

        elif self.action_type in ["HotSpots", "HotSpots2"]:
            self.cb_switches = cb_switches
            hot_spots_buses = self.hot_spots
            self.hot_spots_switch_ids = self.get_hot_spots_switch_ids(hot_spots_buses, self.net)

            if self.action_type == "HotSpots2":
                all_line_switches = set()
                for switches in self.line_switch_dict.values():
                    all_line_switches.update(switches['from_bus'])
                    all_line_switches.update(switches['to_bus'])
                self.hot_spots_switch_ids = all_line_switches.intersection(self.hot_spots_switch_ids)

            self.cb_excluding_ehv = self.get_cbs_excluding_ehv()
            self.cbs_and_hotspot_switch_ids = np.unique(
                np.concatenate([
                    self.cb_excluding_ehv.index.values,
                    np.array(list(self.hot_spots_switch_ids))
                ])
            )

            print("_______ Hot spot switch IDs: _______")
            print(sorted(self.hot_spots_switch_ids))
            print("_______ Combined CBs and hot spot switch IDs: _______")
            print(sorted(self.cbs_and_hotspot_switch_ids))

            action_space = spaces.MultiDiscrete([2] * self.cbs_and_hotspot_switch_ids.shape[0])

            if self.is_train:
                self.env_meta.update({
                    "hot_spots_switch_ids": [int(i) for i in self.hot_spots_switch_ids],
                    "cbs_and_hotspot_switch_ids": [int(i) for i in self.cbs_and_hotspot_switch_ids],
                    "hot_spots": [int(i) for i in self.hot_spots]
                })
                with open("env_meta.json", "w") as file:
                    json.dump(self.env_meta, file, indent=4)
                print("_______ Saved hot spot metadata to env_meta.json _______")


        elif self.action_type in ["EncMask_CBandAllLinesFltrd"]:
            self.cb_excluding_ehv = self.get_cbs_excluding_ehv()
            self.line_ids = list(self.line_switch_dict.keys())

            # Create action space dimensions
            action_dims = []

            # For circuit breakers (excluding EHV): each CB has 2 actions [0, 1]
            num_cbs = len(self.cb_excluding_ehv)
            action_dims.extend([2] * num_cbs)

            # For line switches: each line has 2 positions, each with 4 encoded actions [0, 1, 2, 3]
            # representing (0,0), (0,1), (1,0), (1,1)
            num_lines = len(self.line_ids)
            action_dims.extend([16] * (num_lines))  # 1 positions per line

            # Create MultiDiscrete action space
            action_space = spaces.MultiDiscrete(action_dims)

            # Store mapping information for action decoding
            self.cb_start_idx = 0
            self.cb_end_idx = num_cbs # not inclusive
            self.line_start_idx = num_cbs
            self.line_end_idx = num_cbs + (num_lines)

            # Encoding mapping for line switches
            self.line_encoding_map = {
                  0: (0, 0, 0, 0),
                  1: (0, 0, 0, 1),  #not allowed
                  2: (0, 0, 1, 0),  #not allowed
                  3: (0, 0, 1, 1),  #not allowed
                  4: (0, 1, 0, 0),  #not allowed
                  5: (0, 1, 0, 1), 
                  6: (0, 1, 1, 0), 
                  7: (0, 1, 1, 1), #not allowed
                  8: (1, 0, 0, 0),  #not allowed
                  9: (1, 0, 0, 1),
                  10: (1, 0, 1, 0), 
                  11: (1, 0, 1, 1), #not allowed
                  12: (1, 1, 0, 0), #not allowed
                  13: (1, 1, 0, 1), #not allowed
                  14: (1, 1, 1, 0), #not allowed
                  15: (1, 1, 1, 1) #not allowed
              }

            # 16  (0,0,0,0)  single

            print(f"Action space created:")
            print(f"  - CBs (excluding EHV): {num_cbs} actions [0,1]")
            print(f"  - Line switches: {num_lines} lines × 2 positions × 4 encodings = {num_lines * 2} actions [0,1,2,3]")
            print(f"  - Total action space: {action_dims}")
            print(f"  - CB indices: {self.cb_start_idx} to {self.cb_end_idx}")
            print(f"  - Line indices: {self.line_start_idx} to {self.line_end_idx}")
        else:
            action_space = spaces.MultiDiscrete([2] * self.net.switch.shape[0])

        # Discrete space includes switch states and line states
        num_lines = self.net.line.shape[0]
        discrete_space = MultiDiscrete([2] * self.net.switch.shape[0] + [2] * num_lines)

        # Observation space sizes
        num_bus = self.net.bus.shape[0]
        num_sgen = self.net.sgen.shape[0]
        num_load = self.net.load.shape[0]
        num_ext_grid = self.net.ext_grid.shape[0]
        num_storage = self.net.storage.shape[0]

        # Continuous observation spaces
        observation_space = spaces.Dict({
            "discrete_switches": discrete_space,
            "continuous_vm_bus": Box(low=0.5, high=1.5, shape=(num_bus,), dtype=np.float32),
            "continuous_sgen_data": Box(low=0.0, high=1e5, shape=(num_sgen,), dtype=np.float32),
            "continuous_load_data": Box(low=0.0, high=1e5, shape=(num_load,), dtype=np.float32),
            "continuous_line_loadings": Box(low=0.0, high=800.0, shape=(num_lines,), dtype=np.float32),
            "continuous_space_ext_grid_p_mw": Box(low=-5e7, high=5e7, shape=(num_ext_grid,), dtype=np.float32),
            "continuous_space_ext_grid_q_mvar": Box(low=-5e7, high=5e7, shape=(num_ext_grid,), dtype=np.float32),
            "continuous_storage_data": Box(low=-1e3, high=1e3, shape=(num_storage,), dtype=np.float32),
            "time_features": Box(
                low=np.array([2000, 1, 1, 0, 0,0], dtype=np.float32),  # [min_year, min_month, min_day, min_hour, min_minute, min_day_of_week]
                high=np.array([3000, 12, 31, 23,59,6], dtype=np.float32), # [max_year, max_month, max_day, max_hour, max_minute, max_day_of_week]
                shape=(6,),
                dtype=np.float32
            )
        })

        print("_______ Action space defined with shape: _______")
        print(action_space.shape)
        print("_______ Observation space components: _______")
        for key, space in observation_space.spaces.items():
            print(f"  - {key}: shape={space.shape}, dtype={space.dtype}")

        return action_space, observation_space



    def set_to_all_data(self):
        # self.gen_data = self.gen_data_normalized
        self.load_data = self.load_data_normalized
        self.sgen_data = self.sgen_data_normalized
        self.storage_data = self.storage_data_normalized
        return


    def apply_absolute_values(self, net, absolute_values_dict, case_or_time_step):
        for elm_param in absolute_values_dict.keys():
            if absolute_values_dict[elm_param].shape[1]:
                elm = elm_param[0]
                param = elm_param[1]
                net[elm].loc[:, param] = absolute_values_dict[elm_param].loc[case_or_time_step]
        return net


    def reset(self, options: Optional[Dict[str, Any]] = None,
              seed: Optional[int] = None, ts: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Resets the environment state, including resetting the time step, reapplying profiles,
        and running a load flow calculation. If any errors occur, the environment is terminated and truncated.

        Args:
            options (Optional[Dict[str, Any]]): Optional options dictionary (not used in this function during training). Only used for evaluation.
            seed (Optional[int]): Optional random seed (not used in this function during training). Only used for evaluation.
            ts (Optional[int]): A specific time step to reset to. If None, the next time step is used.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: The current observation and info dictionaries.
        """

        print("============ RESETING =======")


        # Reset environment flags and counters
        self.terminated = False
        self.truncated = False
        self.count = 0

        # Determine the time step to use
        # Determine the time step to use
        self.time_step = ts if ts is not None else self.time_step + 1

        # Check if the time step is out of range and reset it if necessary
        if self.time_step >= self.sgen_data.shape[0]:
            self.time_step = 0  # Reset time step to the first index if out of range

        # Get the relative index for the current time step
        self.relative_index = self.sgen_data.index.values[self.time_step]

        # Print the current time step, relative index, and count for debugging purposes
        print(f"Current Time Step: {self.time_step}")
        print(f"Relative Index (Time): {self.relative_index}")
        print(f"Current Count: {self.count}")


        # Create a deep copy of the initial network and apply absolute values
        self.net = copy.deepcopy(self.initial_net)
        self.net = self.apply_absolute_values(self.net, self.profiles, self.relative_index)

        # Attempt to run load flow and handle errors
        try:
            pp.runpp(self.net)
            print(f"Load flow passed in resetting")
        except Exception as e:
            self.terminated = True
            self.truncated = True
            print(f"Load flow error in resetting: {e}")
            self.convergence_error_count += 1
            print("============ END RESETING =======")
            return self.observation, self.info

        # Check if any lines are disconnected or if there are NaN values in results
        if self.net.res_line['loading_percent'].isna().sum() > self.allowed_lines:
            self.terminated = True
            self.truncated = True
            print("Line disconnect error in resetting")
            self.line_disconnect_count += 1
            print("============ END RESETING =======")

            return self.observation, self.info

        if self.net.res_bus['vm_pu'].isna().any():
            self.terminated = True
            self.truncated = True
            print("Vm pu error in resetting")
            self.nan_vm_pu_count += 1
            print("============ END RESETING =======")
            return self.observation, self.info

        # Extract relevant results from the network
        loading_percent = self.net.res_line['loading_percent'].fillna(0).values.astype(np.float32)
        vm_pu = self.net.res_bus['vm_pu'].fillna(0).values.astype(np.float32)

        # Prepare discrete and continuous observations
        initial_discrete = np.concatenate([
            self.net.switch['closed'].astype(int).values,
            self.net.line['in_service'].astype(int).values
        ])

        # Continuous observations from system data
        initial_continuous_loading = loading_percent
        initial_continuous_vm = vm_pu
        initial_continuous_sgen = self.sgen_data.values[self.time_step].astype(np.float32)
        initial_continuous_load = self.load_data.values[self.time_step].astype(np.float32)
        initial_ext_grid_p_mw = self.net.res_ext_grid['p_mw'].fillna(0).values.astype(np.float32)
        initial_ext_grid_q_mvar = self.net.res_ext_grid['q_mvar'].fillna(0).values.astype(np.float32)
        initial_storage_p_mw = self.net.res_storage['p_mw'].fillna(0).values.astype(np.float32)

        if "time" in self.include_features:
          # Get the current timestamp from the profiles time column
          current_timestamp = self.net.profiles['renewables'].iloc[self.relative_index]['time']
          print(f"Current timestamp: {current_timestamp}")

          # Convert the timestamp to a datetime object
          if isinstance(current_timestamp, str):
              # Parse the timestamp string
              dt = datetime.datetime.strptime(current_timestamp, "%d.%m.%Y %H:%M")
          else:
              # If it's already a datetime object
              dt = current_timestamp

          year = dt.year
          month = dt.month
          day = dt.day
          hour = dt.hour
          minute = dt.minute
          day_of_week = dt.weekday()
          time_features = np.array([
              year,
              month,
              day,
              hour,
              minute,
              day_of_week
          ], dtype=np.float32)

          print(f"Current timestamp: {current_timestamp}")
          print(f"Extracted time features - Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Minute: {minute}, Day of Week: {day_of_week}")
          print(f"Normalized time features: {time_features}")

        else:
          time_features = np.zeros(5, dtype=np.float32)

        # Construct the observation dictionary
        self.observation = {
            "discrete_switches": initial_discrete,
            "continuous_vm_bus": initial_continuous_vm,
            "continuous_sgen_data": initial_continuous_sgen,
            "continuous_load_data": initial_continuous_load,
            "continuous_line_loadings": initial_continuous_loading,
            "continuous_space_ext_grid_p_mw": initial_ext_grid_p_mw,
            "continuous_space_ext_grid_q_mvar": initial_ext_grid_q_mvar,
            "continuous_storage_data": initial_storage_p_mw,
            "time_features": time_features
        }


        print("============ END RESETING =======")

        return self.observation, self.info


    def build_switch_pair_map(self, switch_df: pd.DataFrame) -> Dict[int, int]:
        """
        Build a mapping of switch pairs from a DataFrame.

        For each row in the DataFrame, it maps the switches from 'SW_A_from', 'SW_B_from',
        'SW_A_to', and 'SW_B_to' to form a switch pair map.

        Args:
        - switch_df (pd.DataFrame): A DataFrame containing switch information.

        Returns:
        - Dict[int, int]: A dictionary mapping each switch to its pair.
        """
        switch_pair_map = {}

        # Iterate through the rows to extract the switch pairs
        for _, row in switch_df.iterrows():
            # For the 'from' pair
            sw1 = row['SW_A_from']
            sw2 = row['SW_B_from']
            if pd.notna(sw1) and pd.notna(sw2):
                switch_pair_map[int(sw1)] = int(sw2)
                switch_pair_map[int(sw2)] = int(sw1)

            # For the 'to' pair
            sw3 = row['SW_A_to']
            sw4 = row['SW_B_to']
            if pd.notna(sw3) and pd.notna(sw4):
                switch_pair_map[int(sw3)] = int(sw4)
                switch_pair_map[int(sw4)] = int(sw3)

        # Print the switch pair map for debugging
        print("_______ Switch Pair Map: _______")
        print(switch_pair_map)

        return switch_pair_map

    def check_validity(self, net: pandapowerNet, switch_df: pd.DataFrame) -> int:
        """
        Check the validity of switches based on their states and pair mappings.

        The function counts how many switch pairs are both closed (state = 1),
        which is considered an invalid state.

        Args:
        - net (pandapowerNet): The network object containing the switch states.
        - switch_df (pd.DataFrame): A DataFrame containing the switch information.

        Returns:
        - int: The number of invalid switch pairs (both switches in the pair are closed).
        """
        switch_states = net.switch['closed']  # Retrieve the current switch states
        switch_pair_map = copy.deepcopy(self.switch_pair_map)
        invalid_count = 0

        # Check for invalid switch pairs
        # for sw in self.cbs_and_hotspot_switch_ids:
        for sw in net.switch.index:
            state = switch_states[sw]
            counter_sw = switch_pair_map.get(sw, None)

            if counter_sw is None:  # If no pair exists, skip this switch
                continue

            counter_state = switch_states[counter_sw]

            # If both switches are closed (state = 1), it's an invalid pair
            if state == 1 and counter_state == 1:
                invalid_count += 1

        # Print the invalid count for debugging purposes
        print(f"_______ Invalid Switch Pair Count: {invalid_count} _______")

        return invalid_count


    def step(self, action):
        """
        Takes an action to update the environment state and calculates the reward.

        Args:
            action (list or array): The action taken by the agent, typically representing the state of the switches.

        Returns:
            observation (dict): The updated observation of the environment.
            float: The calculated reward for the action taken.
            bool: Flag indicating whether the episode is terminated.
            bool: Flag indicating whether the episode is truncated.
            dict: Additional information about the current step.
        """
        print("===== STEPPING ======")
        print(f"Action length: {len(action)}")
        #print action
        print("Action: ", action)

        self.info = {}

        if self.terminated:
            print("Reset Needed: Environment is terminated")
            print("============ END STEPPING =======")
            return self.observation, 0, self.terminated, self.truncated, self.info

        max_loading_before = self.net.res_line['loading_percent'].max()
        print(f"Max loading before action: {max_loading_before}")

        self.info["max_loading_before"] = max_loading_before

        if self.action_type == "NodeSplitting":
            print("Action type: NodeSplitting")
            for i, switch_idx in enumerate(self.cb_switches.index):
                self.net.switch.at[switch_idx, 'closed'] = bool(action[i])
        elif self.action_type == "NodeSplittingExEHVCBs":
            print("Action type: NodeSplittingExEHVCBs")
            for i, switch_idx in enumerate(self.cb_excluding_ehv.index):
                self.net.switch.at[switch_idx, 'closed'] = bool(action[i])
        elif self.action_type == "HotSpots":
            print("Action type: HotSpots")
            for i, switch_idx in enumerate(self.cbs_and_hotspot_switch_ids):
                self.net.switch.at[switch_idx, 'closed'] = bool(action[i])
        elif self.action_type == "HotSpots2":
            print("Action type: HotSpots2")
            for i, switch_idx in enumerate(self.cbs_and_hotspot_switch_ids):
                self.net.switch.at[switch_idx, 'closed'] = bool(action[i])
        elif self.action_type == "EncMask_CBandAllLinesFltrd":
            print("Action type: EncMask_CBandAllLinesFltrd")

            # Apply CB actions
            num_cbs = len(self.cb_excluding_ehv)
            for i, switch_idx in enumerate(self.cb_excluding_ehv.index):
                self.net.switch.at[switch_idx, 'closed'] = bool(action[i])

            # Apply line actions
            action_idx = num_cbs  # Start after CB actions

            for line_idx, line_id in enumerate(self.line_ids):
                # Get the row for this line from switch_df
                line_row = self.switch_df[self.switch_df['line_id'] == line_id].iloc[0]
                
                # Get switch IDs for this line
                sw_a_from = int(line_row['SW_A_from'])
                sw_b_from = int(line_row['SW_B_from'])
                sw_a_to = int(line_row['SW_A_to'])
                sw_b_to = int(line_row['SW_B_to'])
                
                # Decode the single action (0-15) to get all 4 switch states
                line_action = action[action_idx]
                state_a_from, state_b_from, state_a_to, state_b_to = self.line_encoding_map[line_action]
                
                # Apply states to switches
                self.net.switch.at[sw_a_from, 'closed'] = bool(state_a_from)
                self.net.switch.at[sw_b_from, 'closed'] = bool(state_b_from)
                self.net.switch.at[sw_a_to, 'closed'] = bool(state_a_to)
                self.net.switch.at[sw_b_to, 'closed'] = bool(state_b_to)
                
                # Move to next line's action (each line has 1 action with 16 possibilities)
                action_idx += 1
                
                # print(f"Line {line_id}: Action {line_action} -> SW_A_from({sw_a_from})={state_a_from}, "
                #       f"SW_B_from({sw_b_from})={state_b_from}, SW_A_to({sw_a_to})={state_a_to}, "
                #       f"SW_B_to({sw_b_to})={state_b_to}")
        else:
            print("Action type: Default (all switches)")
            for i in range(self.net.switch.shape[0]):
                self.net.switch.at[i, 'closed'] = bool(action[i])

        try:
            pp.runpp(self.net)
            print("Load flow passed in stepping")
        except:
            self.terminated = True
            self.truncated = True
            print("Load flow error in stepping")
            self.convergence_error_count += 1
            print("============ END STEPPING =======")
            return self.observation, self.convergence_penalty, self.terminated, self.truncated, self.info

        if self.action_type in ["HotSpots2", "EncMask_CBandAllLinesFltrd"]:
            invalid_count = self.check_validity(self.net, self.switch_df)
            invalid_penalty = self.penalty_scalar * invalid_count
            _penalty = 0
            print(f"Invalid penalty: {invalid_penalty}")

            if self.net.res_bus['vm_pu'].isna().any():
                self.nan_vm_pu_count += 1
                print("Vm pu error in HotSpots2")
                if self.nan_vm_pu_penalty == "dynamic":
                    _penalty = self.penalty_scalar * self.net.res_bus['vm_pu'].isna().sum()
                    print(f"Vm pu dynamic penalty: {_penalty}")

            if invalid_count > 0:
                self.terminated = True
                total_penalty = invalid_penalty + _penalty
                print(f"Total penalty for HotSpots2: {total_penalty}")
                print("============ END STEPPING =======")
                self.info["status"] = "Failed"
                return self.observation, total_penalty, self.terminated, self.truncated, self.info

        if self.net.res_line['loading_percent'].isna().sum() > self.allowed_lines:
            self.terminated = True
            print("Line disconnection error")
            self.line_disconnect_count += 1
            print("============ END STEPPING =======")
            self.info["status"] = "Failed"
            return self.observation, self.line_disconnect_penalty, self.terminated, self.truncated, self.info

        if self.net.res_bus['vm_pu'].isna().any():
            self.terminated = True
            print("Vm pu error")
            self.nan_vm_pu_count += 1
            if self.nan_vm_pu_penalty == "dynamic":
                _penalty = self.penalty_scalar * self.net.res_bus['vm_pu'].isna().sum()
                print(f"Vm pu dynamic penalty: {_penalty}")
                print("============ END STEPPING =======")
                self.info["status"] = "Failed"
                return self.observation, _penalty, self.terminated, self.truncated, self.info
            elif self.nan_vm_pu_penalty == "scaled":
                _penalty = self.penalty_scalar * (self.net.res_bus['vm_pu'].isna().sum() / self.net.bus.shape[0])
                print(f"Vm pu scaled penalty: {_penalty}")
                print("============ END STEPPING =======")
                return self.observation, _penalty, self.terminated, self.truncated, self.info
            else:
                print("============ END STEPPING =======")
                self.info["status"] = "Failed"
                return self.observation, self.nan_vm_pu_penalty, self.terminated, self.truncated, self.info

        print("All systems operational")

        self.net.res_line['loading_percent'] = self.net.res_line['loading_percent'].fillna(0)
        P_j_t = np.array([line['loading_percent'] / 100 for _, line in self.net.res_line.iterrows()])

        max_loading_after = self.net.res_line['loading_percent'].max()
        print(f"Max loading after action: {max_loading_after}")

        self.info["max_loading_after"] = max_loading_after

        R_congestion_t = self.calculate_congestion_reward(P_j_t, max_loading_before, max_loading_after)
        R_t = R_congestion_t

        self.info["R_congestion_t"] = R_congestion_t
        self.info["status"] = "Passed"

        if self.count >= self.max_step:
            self.truncated = True
            self.terminated = True
            print("Max step reached, episode truncated")
            print("============ END STEPPING =======")
            return self.observation, R_t, self.terminated, self.truncated, self.info

        self.observation, flag = self.update_state()
        self.terminated = flag
        self.count += 1
        print(f"Step count: {self.count}")
        print("============ END STEPPING =======")
        return self.observation, R_t, self.terminated, self.truncated, self.info



    def get_data_length(self) -> Tuple[int, int]:
        """
        Returns the lengths of the test and training data.

        This function retrieves the lengths of the test and training datasets
        that were provided during initialization.

        Returns:
        - Tuple[int, int]: A tuple containing the length of the test data
          and the length of the training data.
        """
        return self.test_data_length, self.train_data_length

    def calculate_congestion_reward(self, rho: np.ndarray, max_loading_before: float, max_loading_after: float) -> float:
        """
        Calculates the congestion reward based on the provided rho values (congestion factor) and line loadings.

        Args:
        - rho (np.ndarray): Array of congestion factors for each line.
        - max_loading_before (float): Maximum line loading before action.
        - max_loading_after (float): Maximum line loading after action.

        Returns:
        - float: The calculated congestion reward.
        """
        # Ensure rho values are not below the minimum threshold
        _temp = np.maximum(self.rho_min, rho)

        # Calculate u_t: the sum of adjusted congestion values
        u_t = np.sum(1 - (_temp - self.rho_min))

        # # Debugging prints with detailed information
        # print("==== Congestion Reward Calculation ====")
        # print(f"Rho (congestion factors): {rho}")
        # print(f"Adjusted Rho (after applying rho_min): {_temp}")
        # print(f"Calculated u_t (sum of congestion): {u_t}")
        # print("=======================================")

        # Calculate the bonus based on the loading difference and the reward type
        if self.reward_type == "scaled":
            bonus = self.bonus_constant * (max_loading_before - max_loading_after) / max_loading_before
            u_t = u_t / self.net.line.shape[0]
        elif self.reward_type == "CheckNewReward1":
            R_congestion = -1*sum(rho)
            print(f"R_congestion: {R_congestion}")
            return R_congestion
        else:
            bonus = self.bonus_constant * (max_loading_before - max_loading_after)

        # Debugging prints for bonus and final reward
        print("==== Bonus and Final Reward ====")
        print(f"Bonus calculated: {bonus}")
        print(f"u_t (scaled by number of lines if 'scaled'): {u_t}")
        print(f"Final Congestion Reward (R_congestion): {u_t + bonus}")
        print("==================================")

        # Return the final congestion reward
        R_congestion = u_t + bonus
        return R_congestion

    def update_state(self):
        """
        Updates the state of the environment based on the current time step.
        This includes updating the discrete switches, continuous grid states,
        and performing load flow calculations. If any issues are encountered
        (e.g., load flow error or line disconnects), it returns an error.

        Returns:
            observation (dict): The updated observation after the state has been updated.
            bool: A flag indicating whether the environment has terminated (True) or not (False).
        """
        print("============ UPDATING =======")

        # Increment time step and reset if out of bounds
        self.time_step += 1
        if self.time_step >= self.sgen_data.shape[0]:
            self.time_step = 0  # Reset to 0 if the time step exceeds the data length

        # Log the current time step
        print(f"Updating state for time_step: {self.time_step}")

        # Initialize the network to its original state
        self.net = copy.deepcopy(self.initial_net)

        # Set the relative index for the current time step
        self.relative_index = self.sgen_data.index.values[self.time_step]

        # Apply the absolute values for the current time step
        self.net = self.apply_absolute_values(self.net, self.profiles, self.relative_index)

        # Perform load flow calculations
        try:
            pp.runpp(self.net)
            print("Load flow passed in updating")
        except:
            print("Load flow error in updating")
            self.convergence_error_count += 1
            return self.observation, True  # Return error flag if load flow fails

        # Check for invalid line loading percentages
        loading_percent = self.net.res_line['loading_percent']
        if loading_percent.isna().sum() > self.allowed_lines:
            print("Line disconnect error in updating")
            self.line_disconnect_count += 1
            return self.observation, True  # Return error flag if line disconnect is detected

        # Check for NaN values in bus voltage magnitude (vm_pu)
        if self.net.res_bus['vm_pu'].isna().any():
            print("Vm pu error in updating")
            self.nan_vm_pu_count += 1
            return self.observation, True  # Return error flag if NaN in vm_pu

        # Normalize loading percent and voltage magnitude data
        loading_percent = self.net.res_line['loading_percent'].fillna(0).values.astype(np.float32)
        vm_pu = self.net.res_bus['vm_pu'].fillna(0).values.astype(np.float32)

        # Prepare discrete states (switches and lines)
        initial_discrete = np.concatenate([
            self.net.switch['closed'].astype(int).values,
            self.net.line['in_service'].astype(int).values
        ])

        # Prepare continuous states (grid data)
        initial_continuous_loading = loading_percent
        initial_continuous_vm = vm_pu
        initial_continuous_sgen = self.sgen_data.values[self.time_step].astype(np.float32)
        initial_continuous_load = self.load_data.values[self.time_step].astype(np.float32)
        initial_ext_grid_p_mw = self.net.res_ext_grid['p_mw'].fillna(0).values.astype(np.float32)
        initial_ext_grid_q_mvar = self.net.res_ext_grid['q_mvar'].fillna(0).values.astype(np.float32)
        initial_storage_p_mw = self.net.res_storage['p_mw'].fillna(0).values.astype(np.float32)

        if "time" in self.include_features:
          # Get the current timestamp from the profiles time column
          current_timestamp = self.net.profiles['renewables'].iloc[self.relative_index]['time']
          print(f"Current timestamp: {current_timestamp}")

          # Convert the timestamp to a datetime object
          if isinstance(current_timestamp, str):
              # Parse the timestamp string
              dt = datetime.datetime.strptime(current_timestamp, "%d.%m.%Y %H:%M")
          else:
              # If it's already a datetime object
              dt = current_timestamp

          year = dt.year
          month = dt.month
          day = dt.day
          hour = dt.hour
          minute = dt.minute
          day_of_week = dt.weekday()
          time_features = np.array([
              year,
              month,
              day,
              hour,
              minute,
              day_of_week
          ], dtype=np.float32)

          print(f"Current timestamp: {current_timestamp}")
          print(f"Extracted time features - Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Minute: {minute}, Day of Week: {day_of_week}")
          print(f"Normalized time features: {time_features}")

        else:
          time_features = np.zeros(5, dtype=np.float32)


        # Combine the states into a single observation dictionary
        self.observation = {
            "discrete_switches": initial_discrete,
            "continuous_vm_bus": initial_continuous_vm,
            "continuous_sgen_data": initial_continuous_sgen,
            "continuous_load_data": initial_continuous_load,
            "continuous_line_loadings": initial_continuous_loading,
            "continuous_space_ext_grid_p_mw": initial_ext_grid_p_mw,
            "continuous_space_ext_grid_q_mvar": initial_ext_grid_q_mvar,
            "continuous_storage_data": initial_storage_p_mw,
            "time_features": time_features
        }

        # # Log the updated observation
        # print("===== Observation updated =====")
        # print(f"Discrete switches: {initial_discrete}")
        # print(f"Continuous grid data (vm_pu): {initial_continuous_vm}")
        # print(f"Continuous generator data (sgen): {initial_continuous_sgen}")
        # print(f"Continuous load data: {initial_continuous_load}")
        # print(f"Continuous line loading data: {initial_continuous_loading}")
        # print(f"Continuous ext grid power (P MW): {initial_ext_grid_p_mw}")
        # print(f"Continuous ext grid reactive power (Q MVAR): {initial_ext_grid_q_mvar}")
        # print("===================================")

        print("============ END UPDATING =======")

        # Return the updated observation and flag indicating no error
        return self.observation, False





# Define custom callbacks
class TimeStepLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None
        self.end_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        self.logger.record("start_time", self.start_time)

    def _on_training_end(self):
        self.end_time = time.time()
        self.logger.record("end_time", self.end_time)
        self.logger.record("total_time", self.end_time - self.start_time)
        if hasattr(self.training_env, "get_attr"):
            self.logger.record("total_load_flow_error_count", self.training_env.get_attr("convergence_error_count")[0])
            self.logger.record("total_line_disconnect_error_count", self.training_env.get_attr("line_disconnect_count")[0])
            self.logger.record("total_nan_vm_pu_error_count", self.training_env.get_attr("nan_vm_pu_count")[0])


    def _on_step(self):
        time_elapsed = time.time() - self.start_time
        self.logger.record("time_elapsed", time_elapsed)
        self.logger.record("total_timesteps", self.num_timesteps)
        # Ensure environment metrics are logged
        if hasattr(self.training_env, "get_attr"):
            self.logger.record("load_flow_error_count", self.training_env.get_attr("convergence_error_count")[0])
            self.logger.record("line_disconnect_error_count", self.training_env.get_attr("line_disconnect_count")[0])
            self.logger.record("nan_vm_pu_error_count", self.training_env.get_attr("nan_vm_pu_count")[0])

        return True


def linear_schedule(initial_value):
    """Returns a function that computes a linear decay of the learning rate."""
    def func(progress_remaining):
        return progress_remaining * initial_value  # Linearly decrease LR
    return func

# Custom TQDM callback for tracking progress
class TQDMProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.last_timestep = 0

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", leave=True)
        # If we're resuming training, update the progress bar to the current position
        if self.model.num_timesteps > 0:
            self.pbar.update(self.model.num_timesteps)
            self.last_timestep = self.model.num_timesteps

    def _on_step(self) -> bool:
        # Update progress bar by the difference since last update to avoid duplicate counting
        current_step = self.model.num_timesteps
        self.pbar.update(current_step - self.last_timestep)
        self.last_timestep = current_step
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

def main():
    # Load experiment initialization metadata
    with open("init_meta.json", "r") as file:
        init_meta = json.load(file)

    print(init_meta)

    # Extract metadata
    exp_code = init_meta["exp_code"]
    exp_id = init_meta["exp_id"]
    exp_name = init_meta["exp_name"]
    action_type = init_meta["action_type"]
    grid_env = init_meta["grid_env"]

    # Environment settings
    env_config = {
        "is_train": True,
        "simbench_code": "RHVModV1",
        "case_study": "bc",
        "exp_code": exp_code,
        "is_normalize": False,
        "max_step": 96,
        "allowed_lines": 100,
        "convergence_penalty": -200,
        "line_disconnect_penalty": -200,
        "nan_vm_pu_penalty": "dynamic",
        "rho_min": 0.45,
        "reward_type": "Original",
        "penalty_scalar": -10,
        "bonus_constant": 10,
        "action_type": action_type,
        "include_features": ["time"],
        "shuffle": False,
        "hot_spots": set()
    }

    # Instantiate and wrap the environment
    # monitored_env = Monitor(ENV_MaskableRHV(**env_config))
    # env = DummyVecEnv([lambda: monitored_env])
    env = ENV_MaskableRHV(**env_config)

    # Logging setup
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Training configuration
    training_config_meta = {
        "exp_id": exp_id,
        "exp_code": exp_code,
        "exp_name": exp_name,
        "logdir": logdir,
        "env_name": f"ENV_{grid_env}",
        "policy": "MaskableMultiInputActorCriticPolicy",
        "n_epochs": 10,
        "n_steps": 2048,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "total_timesteps": 1_000_000,
        "initial_learning_rate": 0.0003
    }

    # Save training configuration
    with open("training_config_meta.json", "w") as file:
        json.dump(training_config_meta, file, indent=4)
    print("Data saved to training_config_meta.json")

    # Initialize PPO model
    model = MaskablePPO(
        policy=MaskableMultiInputActorCriticPolicy,
        env=env,
        verbose=0,
        tensorboard_log=logdir,
        n_epochs=training_config_meta["n_epochs"],
        n_steps=training_config_meta["n_steps"],
        batch_size=training_config_meta["batch_size"],
        gamma=training_config_meta["gamma"],
        gae_lambda=training_config_meta["gae_lambda"],
        clip_range=training_config_meta["clip_range"],
        ent_coef=training_config_meta["ent_coef"],
        max_grad_norm=training_config_meta["max_grad_norm"]
    )

    # Create callbacks
    tqdm_callback = TQDMProgressCallback(total_timesteps=training_config_meta["total_timesteps"])
    logging_callback = TimeStepLoggingCallback()
    callback_list = CallbackList([logging_callback, tqdm_callback])

    try:
        # Start training
        model.learn(
            total_timesteps=training_config_meta["total_timesteps"],
            tb_log_name=exp_code,
            reset_num_timesteps=False,
            callback=callback_list
        )
        print("Training completed successfully.")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Cleaning up...")

    finally:
        # Final cleanup
        if tqdm_callback.pbar is not None:
            tqdm_callback.pbar.close()

        # Save model
        print(f"Saving model to '{exp_code}'")
        model.save(exp_code)
        print("Model saved successfully.")


if __name__ == "__main__":
    main()
