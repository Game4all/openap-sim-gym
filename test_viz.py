import math
import gymnasium as gym
import time
import numpy as np
from xp_sim_gym.openap_env import OpenAPNavEnv
from xp_sim_gym.viz_wrapper import OpenAPVizWrapper
from xp_sim_gym.config import PlaneEnvironmentConfig

from stable_baselines3 import PPO

def generate_long_route(lat_0, lon_0, num_segments=20):
    route = [{'lat': lat_0, 'lon': lon_0, 'alt': 10000}]
    curr_lat, curr_lon = lat_0, lon_0
    curr_bng = 45 # Northeast
    
    for _ in range(num_segments):
        dist_nm = np.random.uniform(40, 70)
        turn = np.random.uniform(-30, 30)
        curr_bng = (curr_bng + turn) % 360
        
        d_lat = dist_nm / 60.0 * math.cos(math.radians(curr_bng))
        d_lon = dist_nm / 60.0 * math.sin(math.radians(curr_bng)) / math.cos(math.radians(curr_lat))
        
        curr_lat += d_lat
        curr_lon += d_lon
        route.append({'lat': curr_lat, 'lon': curr_lon, 'alt': 10000})
    return route

def main():
    # 1. Setup config (Stage 3 for strong wind)
    lat_start, lon_start = 48.8566, 2.3522
    long_route = generate_long_route(lat_start, lon_start, num_segments=15)

    config = PlaneEnvironmentConfig(
        aircraft_type="A320",
        initial_lat=lat_start,
        initial_lon=lon_start,
        nominal_route=long_route,
    )
    
    # 2. Create env and wrap it
    env = OpenAPNavEnv(config)
    env.set_pretraining_stage(3)
    env = OpenAPVizWrapper(env)
    
    # 3. Load Model
    model_path = "ppo_flight_deviation_pretrained.zip"
    try:
        model = PPO.load(model_path, env=env)
        print(f"Loaded pretrained model from {model_path}")
        use_model = True
    except Exception as e:
        print(f"Could not load model: {e}. Falling back to random actions.")
        use_model = False

    print(f"Starting simulation test with {len(long_route)} waypoints...")
    obs, info = env.reset()
    
    done = False
    step_count = 0
    
    try:
        while not done and step_count < 300:
            # 4. Get action from model or random
            if use_model:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 5. Render
            env.render()
            
            # Slow down for visibility
            time.sleep(0.5) # Faster now that it's a model
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"Step {step_count}, Reward: {reward:.2f}, XTE: {env.env._calculate_xte():.2f}")
                
        print("Simulation finished.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
