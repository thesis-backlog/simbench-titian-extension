from Environments.Main.ENV_RHV_IEEE_PES import ENV_RHV
from stable_baselines3 import PPO
import time

# Create environment
print("Creating environment...")
env = ENV_RHV(simbench_code='RHVModV1-nominal', max_step=50)

# Create PPO agent
print("Initializing PPO agent...")
model = PPO('MultiInputPolicy', env, verbose=1)

# Train for approximately 10 minutes
print('Starting training for ~10 minutes...')
start_time = time.time()

# Adjust total_timesteps if needed (10000-20000 should take ~10 minutes)
model.learn(total_timesteps=10000)

elapsed_time = time.time() - start_time

print(f'\nTraining completed in {elapsed_time/60:.2f} minutes')

# Save the trained model
model.save('ppo_rhv_test')
print('Model saved as ppo_rhv_test.zip')
