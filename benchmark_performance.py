"""
Script de benchmarking d'une baseline (FMS avec direction magnétique, sans correction de trajectoire) vs modèle de déviation.
"""

import numpy as np
from stable_baselines3 import PPO
from xp_sim_gym.openap_env import OpenAPNavEnv
from xp_sim_gym.config import PlaneEnvironmentConfig
from xp_sim_gym.route_generator import RouteStageGenerator


def run_simulation(env, model=None, description="Simulation"):
    print(f"\n--- Running: {description} ---")
    obs, info = env.reset()
    done = False

    total_reward = 0.0
    total_fuel = 0.0
    total_time = 0.0
    max_xte = 0.0
    steps = 0
    total_progression = 0.0
    total_distance = 0.0

    while not done:
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Baseline: Heading Offset 0.0, Duration 5 min (-1.0)
            action = np.array([0.0, -1.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        total_fuel += info.get("fuel_consumed", 0)
        # Dynamic duration from info
        duration_min = info.get("duration", 5.0)
        total_time += duration_min

        max_xte = max(max_xte, abs(info.get("xte", 0)))
        total_progression += info.get("progression", 0)
        total_distance += info.get("distance_flown", 0)
        steps += 1

        if steps % 5 == 0:
            print(
                f"Step {steps}, XTE: {info.get('xte', 0):.2f} NM, Fuel: {total_fuel:.1f} kg")

    success = env.current_waypoint_idx >= len(env.nominal_route)

    return {
        "reward": total_reward,
        "fuel": total_fuel,
        "time": total_time,
        "max_xte": max_xte,
        "progression": total_progression,
        "distance": total_distance,
        "success": success,
        "steps": steps
    }


def main():
    # 1. Setup Base Config for Stage 5 (Hardest)
    base_config = PlaneEnvironmentConfig(
        aircraft_type="A320",
        initial_lat=48.8566,
        initial_lon=2.3522,
    )

    # 2. Generate a COMPLEX scenario (Route + Wind)
    generator = RouteStageGenerator(base_config)
    route, wind_streams = generator.generate(stage=5)

    # Create fixed config for comparison
    benchmark_config = PlaneEnvironmentConfig(
        aircraft_type="A320",
        initial_lat=route[0]['lat'],
        initial_lon=route[0]['lon'],
        nominal_route=route,
        wind_streams=wind_streams,
        randomize_wind=False,
        max_steps=100
    )

    # 3. Load Model
    model_path = "ppo_flight_deviation_pretrained.zip"
    try:
        model = PPO.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Run Baseline
    env_baseline = OpenAPNavEnv(benchmark_config)
    # Ensure reward math etc is consistent
    env_baseline.set_pretraining_stage(6)
    baseline_stats = run_simulation(
        env_baseline, model=None, description="Baseline (Autopilot Only)")

    # 5. Run Model
    env_model = OpenAPNavEnv(benchmark_config)
    env_model.set_pretraining_stage(6)
    model_stats = run_simulation(
        env_model, model=model, description="Trained Agent (PPO)")

    # 6. Report
    print("\n" + "="*50)
    print("                    BENCHMARK RESULTS")
    print("="*50)
    print(f"{'Metric':<20} | {'Baseline':<15} | {'Model':<15}")
    print("-" * 50)
    print(
        f"{'Success':<20} | {str(baseline_stats['success']):<15} | {str(model_stats['success']):<15}")
    print(
        f"{'Total Reward':<20} | {baseline_stats['reward']:<15.2f} | {model_stats['reward']:<15.2f}")
    print(
        f"{'Total Fuel (kg)':<20} | {baseline_stats['fuel']:<15.2f} | {model_stats['fuel']:<15.2f}")
    print(
        f"{'Total Time (min)':<20} | {baseline_stats['time']:<15.2f} | {model_stats['time']:<15.2f}")
    print(
        f"{'Total Distance (NM)':<20} | {baseline_stats['distance']:<15.2f} | {model_stats['distance']:<15.2f}")
    print(
        f"{'Max XTE (NM)':<20} | {baseline_stats['max_xte']:<15.2f} | {model_stats['max_xte']:<15.2f}")
    print(
        f"{'Fuel/NM (avg)':<20} | {baseline_stats['fuel']/(baseline_stats['distance']+1e-6):<15.4f} | {model_stats['fuel']/(model_stats['distance']+1e-6):<15.4f}")
    print(
        f"{'Steps':<20} | {baseline_stats['steps']:<15} | {model_stats['steps']:<15}")
    print("="*50)


if __name__ == "__main__":
    main()
