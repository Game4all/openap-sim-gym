"""
Monte Carlo benchmarking script (2000 runs)
Baseline (Magnetic heading AP) vs PPO Model
Same route and wind for each pair of runs
"""

import numpy as np
from rich.console import Console
from rich import box
from rich.table import Table
from tqdm.rich import tqdm
from stable_baselines3 import PPO
from openap_sim_gym import OpenAPNavEnv, CriticComparisonWrapper, EnvironmentConfig, PlaneConfig,  RouteStageGenerator, BenchmarkRouteGenerator


def run_simulation(env, model=None):
    """Simulates the behavior of the AP model (with or without residual deviation correction) on the specified route."""
    obs, info = env.reset()
    done = False

    total_reward = 0.0
    total_fuel = 0.0
    total_time = 0.0
    max_xte = 0.0
    total_progression = 0.0
    total_distance = 0.0
    steps = 0

    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Baseline AP: no heading correction, fixed duration
            action = np.array([0.0, -1.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        total_fuel += info.get("fuel_consumed", 0.0)
        total_time += info.get("duration", 5.0)
        total_distance += info.get("distance_flown", 0.0)
        total_progression += info.get("progression", 0.0)
        max_xte = max(max_xte, abs(info.get("xte", 0.0)))

        steps += 1

    success = env.get_wrapper_attr("current_waypoint_idx") >= len(
        env.get_wrapper_attr("nominal_route"))

    return {
        "reward": total_reward,
        "fuel": total_fuel,
        "time": total_time,
        "distance": total_distance,
        "fuel_per_nm": total_fuel / (total_distance + 1e-6),
        "max_xte": max_xte,
        "progression": total_progression,
        "steps": steps,
        "success": success,
    }


# Statistical helpers

def aggregate_stats(stats_list):
    keys = ["fuel", "distance", "time", "fuel_per_nm"]
    means = {k: np.mean([s[k] for s in stats_list]) for k in keys}

    return means


def relative_diff(model_value, baseline_value):
    return (model_value - baseline_value) / (baseline_value + 1e-9)


def compute_relative_diffs(model_stats, baseline_stats, keys):
    rel_diffs = {k: [] for k in keys}

    for m, b in zip(model_stats, baseline_stats):
        for k in keys:
            rel = (m[k] - b[k]) / (b[k] + 1e-9)
            rel_diffs[k].append(rel)

    return rel_diffs


def aggregate_relative_diff_stats(rel_diffs):
    return {
        k: {
            "avg": np.mean(v),
            "med": np.median(v),
            "min": np.max(v),
            "max": np.min(v),
        }
        for k, v in rel_diffs.items()
    }


def main():
    N_RUNS = 2000

    keys = ["fuel", "distance", "time", "fuel_per_nm"]

    plane_config = PlaneConfig(
        aircraft_type="A320",
        initial_lat=48.8566,
        initial_lon=2.3522,
    )
    env_config = EnvironmentConfig()

    model_path = "ppo_flight_deviation_pretrained.zip"
    try:
        model = PPO.load(model_path)
        model.policy.set_training_mode(False)
        print(f"Loaded PPO model from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    baseline_results = []
    model_results = []

    generator = BenchmarkRouteGenerator(plane_config)

    for i in tqdm(range(N_RUNS), desc="Running benchmarks"):
        route, wind_streams = generator.generate()

        # Baseline (AP)
        env_baseline = OpenAPNavEnv(plane_config, env_config)
        env_baseline.set_nominal_route(route)
        env_baseline.set_wind_config(wind_streams)

        baseline_stats = run_simulation(env_baseline)

        # PPO model
        env_model = OpenAPNavEnv(plane_config, env_config)
        env_model.set_nominal_route(route)
        env_model.set_wind_config(wind_streams)
        env_model = CriticComparisonWrapper(env_model, model, 0., gamma=0.99)
        model_stats = run_simulation(env_model, model=model)

        baseline_results.append(baseline_stats)
        model_results.append(model_stats)

    baseline_mean = aggregate_stats(baseline_results)
    model_mean = aggregate_stats(model_results)

    rel_mean = {
        k: relative_diff(model_mean[k], baseline_mean[k])
        for k in keys
    }

    rel_diffs = compute_relative_diffs(
        model_results,
        baseline_results,
        keys
    )

    rel_stats = aggregate_relative_diff_stats(rel_diffs)

    console = Console()

    console.rule("[bold cyan]Benchmark: AP vs PPO Model[/bold cyan]")
    console.print(f"[dim]({N_RUNS} paired runs)[/dim]\n")

    # Count wins (lower fuel_per_nm is better)
    wins = sum(1 for m, b in zip(model_results, baseline_results)
               if m["fuel_per_nm"] < b["fuel_per_nm"])
    win_rate = (wins / N_RUNS) * 100
    console.print(
        f"PPO model outperformed baseline [bold green]{wins}/{N_RUNS}[/bold green] times ([bold]{win_rate:.1f}%[/bold])\n")

    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.SIMPLE_HEAVY,
        expand=False,
    )

    table.add_column("Metric", justify="left", no_wrap=True)
    table.add_column("Min Rel Gain Δ", justify="right")
    table.add_column("Avg Rel Gain Δ", justify="right")
    table.add_column("Median Rel Gain Δ", justify="right")
    table.add_column("Max Rel Gain Δ", justify="right")

    def pct(val: float) -> str:
        color = "green" if val >= 0 else "red"
        return f"[{color}]{val * 100:.2f}%[/{color}]"

    for k in keys:
        table.add_row(
            k,
            pct(-rel_stats[k]["min"]),
            pct(-rel_mean[k]),
            pct(-rel_stats[k]["med"]),
            pct(-rel_stats[k]["max"]),
        )

    console.print(table)


if __name__ == "__main__":
    main()
