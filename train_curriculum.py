import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from xp_sim_gym.openap_env import OpenAPNavEnv

class CurriculumCallback(BaseCallback):
    """
    Custom callback for curriculum learning.
    Increases environment difficulty based on elapsed timesteps.
    Could also be based on mean reward/success rate.
    """
    def __init__(self, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.stage = 1
        # Define stage transitions (timestep thresholds)
        self.thresholds = {
            1: 50_000,   # Switch to Stage 2 after 50k steps
            2: 150_000,  # Switch to Stage 3 after 150k steps (Total)
        }
        
    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        
        desired_stage = self.stage
        
        if self.stage == 1 and current_step > self.thresholds[1]:
            desired_stage = 2
        elif self.stage == 2 and current_step > self.thresholds[2]:
            desired_stage = 3
            
        if desired_stage > self.stage:
            self.stage = desired_stage
            
            # Update all environments (if vectorized)
            # self.training_env is usually a VecEnv
            try:
                self.training_env.env_method("set_pretraining_stage", self.stage)
            except AttributeError:
                # Fallback if not VecEnv or method missing
                if hasattr(self.training_env, "set_pretraining_stage"):
                    self.training_env.set_pretraining_stage(self.stage)
            
            if self.verbose > 0:
                print(f"\n[Curriculum] Advancing to Stage {self.stage} at step {current_step}")
                
        return True

def main():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize Environment
    # Passing no nominal_route so it autocreates random ones
    env = OpenAPNavEnv(aircraft_type='A320')
    
    # Initialize Agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )
    
    # Create Curriculum Callback
    curr_callback = CurriculumCallback(verbose=1)
    
    # Train
    print("Starting Pretraining with Curriculum...")
    total_timesteps = 300000
    model.learn(total_timesteps=total_timesteps, callback=curr_callback)
    
    # Save Model
    save_path = "ppo_flight_deviation_pretrained"
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()
