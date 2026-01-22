"""
Script de visualisation d'un vol avec / sans le modèle de déviation.
"""

import math
import time
import argparse
from typing import Optional
import numpy as np
import gymnasium as gym

from xp_sim_gym.openap_env import OpenAPNavEnv
from xp_sim_gym.viz_wrapper import OpenAPVizWrapper
from xp_sim_gym.config import PlaneEnvironmentConfig
from xp_sim_gym.route_generator import RouteStageGenerator

from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser(
        description="Visualisation d'une trajectoire AP ou modèle PPO"
    )
    parser.add_argument(
        "--mode",
        choices=["ap", "model"],
        default="model",
        help="Trajectoire à afficher : 'ap' (autopilot par défaut) ou 'model' (PPO)"
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=5,
        help="Stage du générateur de route (ex: 1–6)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ppo_flight_deviation_pretrained.zip",
        help="Chemin vers le modèle PPO"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Graine aléatoire a utiliser pour la trajectoire."
    )
    args = parser.parse_args()

    # 1. Setup config
    lat_start, lon_start = 48.8566, 2.3522

    base_config = PlaneEnvironmentConfig(
        aircraft_type="A320",
        initial_lat=lat_start,
        initial_lon=lon_start,
    )

    generator = RouteStageGenerator(base_config, args.seed)
    route, wind_streams = generator.generate(stage=args.stage)

    config = PlaneEnvironmentConfig(
        aircraft_type="A320",
        initial_lat=route[0]['lat'],
        initial_lon=route[0]['lon'],
        nominal_route=route,
        wind_streams=wind_streams,
        randomize_wind=False
    )

    # 2. Create env and wrap it
    env = OpenAPNavEnv(config)
    env.set_pretraining_stage(args.stage)
    env = OpenAPVizWrapper(env)

    # 3. Load model if needed
    use_model = False
    model = None

    if args.mode == "model":
        try:
            model = PPO.load(args.model_path, env=env)
            print(f"Loaded pretrained model from {args.model_path}")
            use_model = True
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Falling back to AP trajectory.")
            use_model = False

    print(
        f"Starting simulation | mode={args.mode} | stage={args.stage} "
        f"| waypoints={len(route)}"
    )

    obs, info = env.reset()
    done = False
    step_count = 0

    try:
        while not done and step_count < 300:
            # 4. Action selection
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # AP / baseline behavior (usually "do nothing")
                action = np.zeros(env.action_space.shape, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 5. Render
            env.render()
            time.sleep(0.5)

            step_count += 1
            if step_count % 10 == 0:
                print(
                    f"Step {step_count:03d} | "
                    f"Reward: {reward:6.2f} | "
                    f"XTE: {env.env._calculate_xte():6.2f}"
                )

        print("Simulation finished.")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
