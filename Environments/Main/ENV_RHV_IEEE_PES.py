import gymnasium
import numpy as np
import os
import datetime
import time
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.spaces import MultiDiscrete, Box
import simbench as sb
import copy
from gymnasium import spaces
import pandapower as pp
import warnings
import json

warnings.simplefilter(action='ignore', category=FutureWarning)
class ENV_RHV(gymnasium.Env): # with enhanced observation state
    def __init__(self,
                 simbench_code="1-HV-mixed--0-sw",
                 case_study= 'bc',
                 is_train = True,
                 is_normalize = False,
                 max_step = 50,
                 allowed_lines = 200,
                 convergence_penalty = -200,
                 line_disconnect_penalty = -200,
                 nan_vm_pu_penalty = -200,
                 rho_min = 0.45,
                 action_type = 'NodeSplittingExEHVCBs',
                 exp_code = None,
                 penalty_scalar = -10,
                 bonus_constant = 10,
                 ):
        super().__init__()


        self.simbench_code = simbench_code
        self.net = self.load_simbench_net()
        self.case_study = case_study
        self.is_train = is_train
        self.is_normalize = is_normalize
        self.max_step=max_step

        self.initial_net = None
        self.relative_index = None
        self.time_step = -1

        self.observation = None
        self.action_type = action_type
        self.bonus_constant = bonus_constant

        self.profiles = None
        self.gen_data = None
        self.load_data = None
        self.sgen_data = None

        self.truncated = False
        self.terminated = False
        self.info = dict()

        self.test_data_length = None
        self.train_data_length = None
        self.override_timestep = None
        self.count=0

        self.allowed_lines = allowed_lines
        self.convergence_penalty = convergence_penalty
        self.line_disconnect_penalty = line_disconnect_penalty
        self.nan_vm_pu_penalty = nan_vm_pu_penalty

        self.rho_min = rho_min
        self.line_disconnect_count = 0
        self.convergence_error_count = 0
        self.nan_vm_pu_count = 0

        self.exp_code = exp_code
        self.penalty_scalar = penalty_scalar


        self.initial_net = self.set_study_case(case_study, self.is_train, load_all = True)

        self.action_space , self.observation_space = self.create_act_obs_space()

        _ = self.reset()

        # Define constants and parameters
        self.gamma = 0.99  # Discount factor
        self.rho_max = 1.0  # Maximum acceptable load rate



    def load_simbench_net(self):
        if "nominal" in self.simbench_code:
          net = pp.from_json(f"{self.simbench_code}.json")
        else:
          net= sb.get_simbench_net(self.simbench_code)
        return net

    def set_study_case(self,case_study,is_train, load_all = True):

        if load_all:
          print("Init")
          self.case_study = case_study
          loadcases = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=False)
          net = self.apply_absolute_values(self.net, loadcases, self.case_study)
          self.profiles = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=True)

          # Normalize to the range [0, 1]
          # gen_data_raw = self.profiles[('gen', 'p_mw')]
          load_data_raw = self.profiles[('load', 'p_mw')]
          sgen_data_raw = self.profiles[('sgen', 'p_mw')]
          # self.gen_data_normalized = gen_data_raw.fillna(0)
          self.load_data_normalized = load_data_raw.fillna(0)
          self.sgen_data_normalized = sgen_data_raw.fillna(0)

          # Split into train and test sets (80% train, 20% test)
          # gen_train, gen_test = train_test_split(self.gen_data_normalized, test_size=0.2, shuffle=False)
          load_train, load_test = train_test_split(self.load_data_normalized, test_size=0.2, shuffle=False)
          sgen_train, sgen_test = train_test_split(self.sgen_data_normalized, test_size=0.2, shuffle=False)


          self.test_data_length = sgen_test.shape[0]
          self.train_data_length = sgen_train.shape[0]


          # Based on the train flag, use the appropriate data
          if is_train:
              self.env_meta = {}
              self.env_meta.update({
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
                  "train_data_length": self.train_data_length,
                  "test_data_length": self.test_data_length,
                  # Store the shapes of each dataset
                  # "gen_data_shape": self.gen_data_normalized.shape,
                  "load_data_shape": self.load_data_normalized.shape,
                  "sgen_data_shape": self.sgen_data_normalized.shape,
                  # Store training and testing data shapes
                  # "gen_train_data_shape": gen_train.shape,
                  "load_train_data_shape": load_train.shape,
                  "sgen_train_data_shape": sgen_train.shape,
                  # "gen_test_data_shape": gen_test.shape,
                  "load_test_data_shape": load_test.shape,
                  "sgen_test_data_shape": sgen_test.shape,
                  "exp_code": self.exp_code,
                  "action_type": self.action_type,
                  "penalty_scalar": self.penalty_scalar,
                  "total_CBs" : self.net.switch[(self.net.switch['et'] == 'b') & (self.net.switch['type'] == 'CB')].shape[0],
                  "total_switches": self.net.switch.shape[0],
                  "bonus_constant": self.bonus_constant,
                  "num_cbs_excluding_EHVCBs": self.get_cbs_excluding_ehv().shape[0]
              })

              #save to the env_meta.json
              with open("env_meta.json", "w") as file:
                  json.dump(self.env_meta, file, indent=4)
              print("Data saved to env_meta.json")

              print(self.env_meta)


              # self.gen_data = gen_train
              self.load_data = load_train
              self.sgen_data = sgen_train
          else:
              # print("i am test")
              # self.gen_data = gen_test
              self.load_data = load_test
              self.sgen_data = sgen_test

        return net

    def get_cbs_excluding_ehv(self):
        # Filter switches that are CBs
        cb_switches = self.net.switch[(self.net.switch['et'] == 'b') & (self.net.switch['type'] == 'CB')]

        # Function to check if a bus has a voltage not equal to 220 or 380
        def is_valid_bus(bus_id):
            bus_vn_kv = self.net.bus.loc[bus_id, 'vn_kv']
            return bus_vn_kv not in [220, 380]

        # Filter CBs based on bus and element voltage conditions
        valid_cbs = cb_switches[
            cb_switches.apply(
                lambda row: is_valid_bus(row['bus']) and is_valid_bus(row['element']), axis=1
            )
        ]

        return valid_cbs

    def create_act_obs_space(self):

        if self.action_type == "NodeSplitting":
          self.cb_switches = self.net.switch[(self.net.switch['et'] == 'b') & (self.net.switch['type'] == 'CB')]
          action_space = spaces.MultiDiscrete([2] * self.cb_switches.shape[0])
        elif self.action_type == "NodeSplittingExEHVCBs":
          self.cb_switches = self.net.switch[(self.net.switch['et'] == 'b') & (self.net.switch['type'] == 'CB')]
          self.cb_excluding_ehv = self.get_cbs_excluding_ehv()
          action_space = spaces.MultiDiscrete([2] * self.cb_excluding_ehv.shape[0])
        else:
          action_space = spaces.MultiDiscrete([2] * self.net.switch.shape[0])


        num_lines =  self.net.line.shape[0]  # status of the lines
        discrete_space = MultiDiscrete([2] *  self.net.switch.shape[0]   + [2] * num_lines )  # Each switch can be 0 or 1
        num_line_loadings =  self.net.line.shape[0]   # loading information
        num_bus = self.net.bus.shape[0] # status  of bus
        # num_generators = self.net.gen.shape[0]
        num_sgenerators = self.net.sgen.shape[0]
        num_loads = self.net.load.shape[0]
        num_ext_grid = self.net.ext_grid.shape[0]

        obs_space_size = num_line_loadings + num_bus  + num_sgenerators + num_loads + num_ext_grid*2

        # Define the continuous space (4 grid elements between 0 and 1)  Sgen, gen, load, ext_grid


        # Define the continuous space
        continuous_space_line_loadings = Box(low=0.0, high=800.0, shape=(num_line_loadings,), dtype=np.float32)
        continuous_space_vm_bus = Box(low=0.5, high=1.5, shape=(num_bus,), dtype=np.float32)
        # continuous_space_gen_data = Box(low=0.0, high=100000, shape=(num_generators,), dtype=np.float32)
        continuous_space_sgen_data = Box(low=0.0, high=100000, shape=(num_sgenerators,), dtype=np.float32)
        continuous_space_load_data = Box(low=0.0, high=100000, shape=(num_loads,), dtype=np.float32)
        continuous_space_ext_grid_p_mw = Box(low=-50000000, high=50000000, shape=(num_ext_grid,), dtype=np.float32)
        continuous_space_ext_grid_q_mvar = Box(low=-50000000, high=50000000, shape=(num_ext_grid,), dtype=np.float32)



        # continuous_space = Box(low=0.0, high=1.0, shape=(obs_space_size,), dtype=np.float32) ## ERROR
        # Combine the spaces into a single observation space
        observation_space = spaces.Dict({
            "discrete_switches": discrete_space,
            "continuous_vm_bus": continuous_space_vm_bus,
            # "continuous_gen_data": continuous_space_gen_data,
            "continuous_sgen_data": continuous_space_sgen_data,
            "continuous_load_data": continuous_space_load_data,
            "continuous_line_loadings": continuous_space_line_loadings,
            "continuous_space_ext_grid_p_mw":continuous_space_ext_grid_p_mw,
            "continuous_space_ext_grid_q_mvar":continuous_space_ext_grid_q_mvar
        })

        return action_space, observation_space


    def set_to_all_data(self):
        # self.gen_data = self.gen_data_normalized
        self.load_data = self.load_data_normalized
        self.sgen_data = self.sgen_data_normalized
        return


    def apply_absolute_values(self, net, absolute_values_dict, case_or_time_step):
        for elm_param in absolute_values_dict.keys():
            if absolute_values_dict[elm_param].shape[1]:
                elm = elm_param[0]
                param = elm_param[1]
                net[elm].loc[:, param] = absolute_values_dict[elm_param].loc[case_or_time_step]
        return net


    def reset(self, options=None, seed=None, ts=None):
        self.terminated = False
        self.truncated = False
        self.count=0

        # If timestep is not provided, use a random timestep based on available data
        if ts is None:
            self.time_step += 1
            if self.time_step >= self.sgen_data.shape[0]:
                self.time_step = 0
                self.relative_index = self.sgen_data.index.values[self.time_step]
            else:
                self.relative_index = self.sgen_data.index.values[self.time_step]
        else:
            self.time_step = ts
            self.relative_index = self.sgen_data.index.values[self.time_step]

        self.net = copy.deepcopy(self.initial_net)

        self.net = self.apply_absolute_values(self.net, self.profiles, self.relative_index)

        # Run load flow calculations
        try:
            pp.runpp(self.net)
        except:
            self.terminated = True
            self.truncated = True
            print("Load flow error in resetting")
            self.convergence_error_count += 1
            return self.observation, self.info


        if self.net.res_line['loading_percent'].isna().sum() > self.allowed_lines:
            self.terminated = True
            self.truncated = True
            print("Line diconnect error in resetting")
            self.line_disconnect_count += 1
            return self.observation, self.info

        if self.net.res_bus['vm_pu'].isna().any():
            self.terminated = True
            self.truncated = True
            print("Vm pu error in resetting")
            self.nan_vm_pu_count += 1
            return self.observation, self.info

        loading_percent = self.net.res_line['loading_percent'].fillna(0).values.astype(np.float32)

        vm_pu = self.net.res_bus['vm_pu'].fillna(0).values.astype(np.float32)


        initial_discrete = np.concatenate([
            self.net.switch['closed'].astype(int).values,
            self.net.line['in_service'].astype(int).values
        ])

        initial_continuous_loading = loading_percent
        initial_continuous_vm = vm_pu
        # initial_continuous_gen = self.gen_data.values[self.time_step].astype(np.float32)
        initial_continuous_sgen = self.sgen_data.values[self.time_step].astype(np.float32)
        initial_continuous_load = self.load_data.values[self.time_step].astype(np.float32)
        initial_ext_grid_p_mw = self.net.res_ext_grid['p_mw'].fillna(0).values.astype(np.float32)
        initial_ext_grid_q_mvar = self.net.res_ext_grid['q_mvar'].fillna(0).values.astype(np.float32)



        self.observation = {  "discrete_switches": initial_discrete,
            "continuous_vm_bus": initial_continuous_vm,
            # "continuous_gen_data": initial_continuous_gen,
            "continuous_sgen_data": initial_continuous_sgen,
            "continuous_load_data": initial_continuous_load,
            "continuous_line_loadings": initial_continuous_loading,
            "continuous_space_ext_grid_p_mw":initial_ext_grid_p_mw,
            "continuous_space_ext_grid_q_mvar":initial_ext_grid_q_mvar
        }


        return self.observation, self.info



    def step(self, action):

        if self.terminated:
          print("Reset Needed")
          return self.observation, 0, self.terminated, self.truncated,  self.info

        # Update CB states based on action only for the filtered CB switches
        # for i, switch_idx in enumerate(self.cb_switches.index):
        #     self.net.switch.at[switch_idx, 'closed'] = bool(action[i])

        max_loading_before = self.net.res_line['loading_percent'].max()
        print(f"Max loading before: {max_loading_before}")


        if self.action_type == "NodeSplitting":
          # Update CB states based on action only for the filtered CB switches
          for i, switch_idx in enumerate(self.cb_switches.index):
              self.net.switch.at[switch_idx, 'closed'] = bool(action[i])
        elif self.action_type == "NodeSplittingExEHVCBs":
          print("Action type is NodeSplittingExEHVCBs")
          # Update CB states based on action only for the filtered CB switches
          for i, switch_idx in enumerate(self.cb_excluding_ehv.index):
              self.net.switch.at[switch_idx, 'closed'] = bool(action[i])
        else:
          # for all switches
          for i in range(self.net.switch.shape[0]):
            self.net.switch.at[i, 'closed'] = bool(action[i])

        # Run load flow calculations
        try:
            pp.runpp(self.net)
            print("Load flow passed in stepping")
        except:
            self.terminated = True
            self.truncated = True
            print("Load flow error in stepping")
            self.convergence_error_count += 1
            return self.observation, self.convergence_penalty, self.terminated, self.truncated,  self.info

        # Adding the line disconnection penalty for diconnecting more than allowed lines
        if self.net.res_line['loading_percent'].isna().sum() > self.allowed_lines:
            self.terminated = True
            print("Line diconnect error")
            self.line_disconnect_count += 1
            return self.observation, self.line_disconnect_penalty, self.terminated, self.truncated,  self.info

        # Adding the making vm_pu nan penalty
        if self.net.res_bus['vm_pu'].isna().any():
            self.terminated = True
            print("Vm pu error")
            self.nan_vm_pu_count += 1
            if self.nan_vm_pu_penalty == "dynamic":
              _penalty = self.penalty_scalar * self.net.res_bus['vm_pu'].isna().sum()
              print(f"Penalty: {_penalty}")
              return self.observation, _penalty, self.terminated, self.truncated,  self.info
            else:
              return self.observation, self.nan_vm_pu_penalty, self.terminated, self.truncated,  self.info


        print("ALL WORKING")
        # Extract load rates (rho), voltages, and power losses from pandapower results
        self.net.res_line['loading_percent'] = self.net.res_line['loading_percent'].fillna(0)
        # Compute P_j_t
        P_j_t = np.array([line['loading_percent'] / 100 for _, line in self.net.res_line.iterrows()])


        max_loading_after = self.net.res_line['loading_percent'].max()
        print(f"Max loading after: {max_loading_after}")

        # Calculate rewards
        R_congestion_t = self.calculate_congestion_reward(P_j_t, max_loading_before, max_loading_after)

        # Calculate combined reward with adaptive weights
        R_t = R_congestion_t

        if self.count>=self.max_step:
          self.truncated = True
          self.terminated = True
          return self.observation, R_t, self.terminated, self.truncated , self.info

        self.observation, flag = self.update_state()

        self.terminated = flag

        self.count+=1
        print('count=',self.count)
        return self.observation, R_t, self.terminated, self.truncated , self.info

    def get_data_length(self):
        return self.test_data_length, self.train_data_length

    def calculate_congestion_reward(self, rho, max_loading_before, max_loading_after):
        _temp = np.zeros(len(rho))
        for i in range(len(rho)):
            _temp[i] = np.max([self.rho_min, rho[i]])
        u_t = np.sum(1-(_temp - self.rho_min))
        print(f"Congestion: {u_t}")  # Debugging line

        bonus = self.bonus_constant * (max_loading_before - max_loading_after)
        print(f"Bonus: {bonus}")  # Debugging line

        R_congestion = u_t + bonus
        print(f"R_congestion: {R_congestion}")  # Debugging line
        return R_congestion

    def update_state(self):
        # Update the state based on the current time step
        self.time_step += 1
        # Update the state of the discrete switches and continuous grid
        if self.time_step >= self.sgen_data.shape[0]:
            self.time_step = 0

        self.net = copy.deepcopy(self.initial_net)

        self.relative_index = self.sgen_data.index.values[self.time_step]


        self.net = self.apply_absolute_values(self.net, self.profiles, self.relative_index)

        # Run load flow calculations
        try:
            pp.runpp(self.net)
            print("Load flow passed in updating")
        except:
            print("Load flow error in updating")
            self.convergence_error_count += 1
            return self.observation, True

        # Normalize res_line['loading_percent']
        loading_percent = self.net.res_line['loading_percent']
        if loading_percent.isna().sum() > self.allowed_lines:
            print("Line disconnect error in updating")
            self.line_disconnect_count += 1
            return self.observation, True

        if self.net.res_bus['vm_pu'].isna().any():
            print("Vm pu error in updating")
            self.nan_vm_pu_count += 1
            return self.observation, True

        loading_percent = self.net.res_line['loading_percent'].fillna(0).values.astype(np.float32)
        vm_pu = self.net.res_bus['vm_pu'].fillna(0).values.astype(np.float32)

        initial_discrete = np.concatenate([
            self.net.switch['closed'].astype(int).values,
            self.net.line['in_service'].astype(int).values
        ])


        initial_continuous_loading = loading_percent
        initial_continuous_vm = vm_pu
        # initial_continuous_gen = self.gen_data.values[self.time_step].astype(np.float32)
        initial_continuous_sgen = self.sgen_data.values[self.time_step].astype(np.float32)
        initial_continuous_load = self.load_data.values[self.time_step].astype(np.float32)
        initial_ext_grid_p_mw = self.net.res_ext_grid['p_mw'].fillna(0).values.astype(np.float32)
        initial_ext_grid_q_mvar = self.net.res_ext_grid['q_mvar'].fillna(0).values.astype(np.float32)


        self.observation = {  "discrete_switches": initial_discrete,
            "continuous_vm_bus": initial_continuous_vm,
            # "continuous_gen_data": initial_continuous_gen,
            "continuous_sgen_data": initial_continuous_sgen,
            "continuous_load_data": initial_continuous_load,
            "continuous_line_loadings": initial_continuous_loading,
            "continuous_space_ext_grid_p_mw":initial_ext_grid_p_mw,
            "continuous_space_ext_grid_q_mvar":initial_ext_grid_q_mvar
        }



        return self.observation, False



import json
import os
import datetime
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

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
    #load_init_meata .json file
    with open("init_meta.json", "r") as file:
        init_meta = json.load(file)

    print(init_meta)
    exp_code = init_meta["exp_code"]
    exp_id = init_meta["exp_id"]
    exp_name = init_meta["exp_name"]
    action_type = init_meta["action_type"]
    grid_env = init_meta["grid_env"]
    simbench_code="1-HV-mixed--0-sw"
    case_study= 'bc'
    is_train = True
    is_normalize = False
    max_step = 50
    allowed_lines = 100
    convergence_penalty = -200
    line_disconnect_penalty = -200
    nan_vm_pu_penalty = "dynamic"
    rho_min = 0.45
    n_epochs = 10
    n_steps = 2048
    batch_size = 256
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef = 0.01
    max_grad_norm = 0.5
    total_timesteps = 1_000_000
    penalty_scalar = -10
    bonus_constant = 10
    initial_learning_rate = 0.0003 # Initial LR

    # Create a monitored environment and wrap it in a vectorized environment
    monitored_env = Monitor(ENV_RHV(is_train=is_train, exp_code=exp_code, simbench_code=simbench_code,
                 case_study= case_study,
                 is_normalize = is_normalize,
                 max_step = max_step,
                 allowed_lines = allowed_lines,
                 convergence_penalty = convergence_penalty,
                 line_disconnect_penalty = line_disconnect_penalty,
                 nan_vm_pu_penalty = nan_vm_pu_penalty,
                 rho_min = rho_min,
                 action_type=action_type, penalty_scalar=penalty_scalar,bonus_constant=bonus_constant))
    env = DummyVecEnv([lambda: monitored_env])

    # Define the logging directory
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Define the training meta file
    training_config_meta = {}
    training_config_meta["exp_id"] = exp_id
    training_config_meta["exp_code"] = exp_code
    training_config_meta["exp_name"] = exp_name
    training_config_meta["logdir"] = logdir
    training_config_meta["env_name"] = f"ENV_{grid_env}"
    policy = "MultiInputPolicy"
    ## add to training_config_meta
    training_config_meta["policy"] = policy
    training_config_meta["n_epochs"] = n_epochs
    training_config_meta["n_steps"] = n_steps
    training_config_meta["batch_size"] = batch_size
    training_config_meta["gamma"] = gamma
    training_config_meta["gae_lambda"] = gae_lambda
    training_config_meta["clip_range"] = clip_range
    training_config_meta["ent_coef"] = ent_coef
    training_config_meta["max_grad_norm"] = max_grad_norm
    training_config_meta["total_timesteps"] = total_timesteps
    training_config_meta["initial_learning_rate"] = initial_learning_rate

    #save the training_config_meta
    with open("training_config_meta.json", "w") as file:
        json.dump(training_config_meta, file, indent=4)
    print("Data saved to training_config_meta.json")

    # Initialize the model
    model = PPO(policy, env, verbose=0, # Set verbose to 0 to avoid conflicts with tqdm
               tensorboard_log=logdir,
               n_epochs=n_epochs,
               n_steps=n_steps,
               batch_size=batch_size,
               gamma=gamma,
               gae_lambda=gae_lambda,
               clip_range=clip_range,
               ent_coef=ent_coef,
               max_grad_norm=max_grad_norm)

    # Create the tqdm callback
    tqdm_callback = TQDMProgressCallback(total_timesteps=total_timesteps)

    # Create a callback list with both your original callback and the tqdm callback
    callback_list = CallbackList([TimeStepLoggingCallback(), tqdm_callback])

    try:
        # Learn with our callback list
        model.learn(total_timesteps=total_timesteps,
                   tb_log_name=f"{exp_code}",
                   reset_num_timesteps=False,
                   callback=callback_list)

        print("Training Ends")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Closing progress bar and saving model...")
    finally:
        # Ensure tqdm progress bar is closed properly
        if hasattr(tqdm_callback, 'pbar') and tqdm_callback.pbar is not None:
            tqdm_callback.pbar.close()

        # Save the model even if interrupted
        print(f"Saving model to {exp_code}")
        model.save(f"{exp_code}")
        print("Model saved successfully!")

if __name__ == "__main__":
    main()