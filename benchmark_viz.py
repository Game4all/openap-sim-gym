"""
Script de benchmarking visuel sur une trajectoire du baseline (FMS Heading Magnétique) et du modèle (FMS + offset de trajectoire)
"""

import math
import time
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from xp_sim_gym.openap_env import OpenAPNavEnv
from xp_sim_gym.viz_wrapper import MultiEnvViz
from xp_sim_gym.config import EnvironmentConfig, PlaneConfig
from xp_sim_gym.route_generator import RouteStageGenerator, BenchmarkRouteGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AP vs Model on a single visualization")
    parser.add_argument("--stage", type=int, default=5,
                        help="Stage du générateur de route (1-6)")
    parser.add_argument("--model-path", type=str,
                        default="ppo_flight_deviation_pretrained.zip", help="Chemin vers le modèle PPO")
    parser.add_argument("--seed", type=int, help="Graine aléatoire")
    args = parser.parse_args()

    # 1. Setup shared configuration
    lat_start, lon_start = 48.8566, 2.3522
    plane_config_ap = PlaneConfig(
        aircraft_type="A320", initial_lat=lat_start, initial_lon=lon_start)
    plane_config_model = PlaneConfig(
        aircraft_type="A320", initial_lat=lat_start, initial_lon=lon_start)
    env_config = EnvironmentConfig()

    generator = BenchmarkRouteGenerator(plane_config_ap, args.seed)
    route, wind_streams = generator.generate()

    # 2. Create Envs
    env_ap = OpenAPNavEnv(plane_config_ap, env_config)
    env_ap.set_nominal_route(route)
    env_ap.set_wind_config(wind_streams)

    env_model = OpenAPNavEnv(plane_config_model, env_config)
    env_model.set_nominal_route(route)
    env_model.set_wind_config(wind_streams)

    # 3. Load Model
    model = None
    try:
        model = PPO.load(args.model_path)
        model.policy.set_training_mode(False)
        print(f"Loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")

    # 4. Setup Multi-Env Visualization
    viz = MultiEnvViz(
        envs=[env_ap, env_model],
        labels=["Autopilot", "PPO Model"],
        colors=[(255, 255, 0), (0, 255, 255)]  # Yellow for AP, Cyan for Model
    )

    # 5. Reset Envs
    obs_ap, _ = env_ap.reset()
    obs_model, _ = env_model.reset()

    done_ap = False
    done_model = False
    step_count = 0

    print(f"Starting benchmark visualization | stage={args.stage}")

    try:
        # Time scale: how many real seconds to wait per simulation minute
        time_scale = 0.02

        while not (done_ap and done_model) and step_count < 1000:
            current_durations = [0.0, 0.0]

            # 1. Step Autopilot if not done
            if not done_ap:
                action_ap = np.zeros(
                    env_ap.action_space.shape, dtype=np.float32)
                obs_ap, r_ap, term_ap, trunc_ap, info_ap = env_ap.step(
                    action_ap)
                viz.update_env_info(0, info_ap)
                current_durations[0] = info_ap.get("duration", 5.0)
                done_ap = term_ap or trunc_ap

            # 2. Step PPO Model if not done
            if not done_model:
                if model:
                    action_model, _ = model.predict(
                        obs_model, deterministic=True)
                else:
                    action_model = np.zeros(
                        env_model.action_space.shape, dtype=np.float32)

                obs_model, r_model, term_model, trunc_model, info_model = env_model.step(
                    action_model)
                viz.update_env_info(1, info_model)
                current_durations[1] = info_model.get("duration", 5.0)
                done_model = term_model or trunc_model

            # 3. Render
            viz.render()
            time.sleep(0.5)

            step_count += 1
            if step_count % 20 == 0:
                def get_stats(i):
                    xte = viz.xte_history[i]
                    gs = viz.gs_history[i]
                    if not xte:
                        return "N/A"
                    return (f"XTE: {min(xte):.1f}/{sum(xte)/len(xte):.1f}/{max(xte):.1f} | "
                            f"GS: {int(min(gs))}/{int(sum(gs)/len(gs))}/{int(max(gs))}")

                print(
                    f"Step {step_count:3} | AP: {viz.total_durations[0]:5.1f}m ({get_stats(0)})")
                print(
                    f"         | MD: {viz.total_durations[1]:5.1f}m ({get_stats(1)})")

        time.sleep(10)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        viz.close()


if __name__ == "__main__":
    main()
