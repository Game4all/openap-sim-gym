"""
Script pour calibrer les seuils pour le wrapper critique avec optimisation via Optuna.
"""

import numpy as np
import argparse
import optuna
from rich.console import Console
from rich.table import Table
from rich import box
from rich.progress import track
from stable_baselines3 import PPO
from xp_sim_gym import OpenAPNavEnv, CriticComparisonWrapper, EnvironmentConfig, PlaneConfig, BenchmarkRouteGenerator

import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_simulation(env, model=None):
    obs, info = env.reset()
    done = False

    total_fuel = 0.0
    total_distance = 0.0

    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.array([0.0, -1.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_fuel += info.get("fuel_consumed", 0.0)
        total_distance += info.get("distance_flown", 0.0)

    return {
        "fuel_per_nm": total_fuel / (total_distance + 1e-6)
    }


class OptimizerObjective:
    """
    Class to handle the optimization context (Model, Routes, Baselines)
    so they aren't re-loaded/re-calculated for every Optuna trial.
    """

    def __init__(self, model, routes_data, baseline_rates, plane_config, env_config, weights):
        self.model = model
        self.routes_data = routes_data
        self.baseline_rates = baseline_rates
        self.plane_config = plane_config
        self.env_config = env_config
        self.weights = weights

    def __call__(self, trial):
        threshold = trial.suggest_float("threshold", 0.0, 0.5)
        gamma = trial.suggest_categorical(
            "gamma", [0.90, 0.95, 0.98, 0.99, 0.995, 0.999])

        config_improvements = []


        for i, (route, wind) in enumerate(self.routes_data):
            env = OpenAPNavEnv(self.plane_config, self.env_config)
            env.set_nominal_route(route)
            env.set_wind_config(wind)

            wrapped_env = CriticComparisonWrapper(
                env, self.model, threshold=threshold, gamma=gamma)

            stats = run_simulation(wrapped_env, model=self.model)
            run_rate = stats["fuel_per_nm"]
            baseline_rate = self.baseline_rates[i]

            imp = (baseline_rate - run_rate) / (baseline_rate + 1e-9)
            config_improvements.append(imp)

        mean_imp = np.mean(config_improvements) * 100
        worst_case_imp = np.min(config_improvements) * 100
        best_case_imp = np.max(config_improvements) * 100

        trial.set_user_attr("mean_imp", mean_imp)
        trial.set_user_attr("worst_case", worst_case_imp)
        trial.set_user_attr("best_case", best_case_imp)

        score = (self.weights['mean'] * mean_imp) + \
                (self.weights['worst'] * worst_case_imp) + \
                (self.weights['best'] * best_case_imp)

        return score


def main():
    parser = argparse.ArgumentParser(
        description="Optuna Optimization for CriticComparisonWrapper threshold.")

    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials to run")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of routes to test per trial (Batch size)")

    parser.add_argument("--w-mean", type=float, default=1.0,
                        help="Weight for Mean Gain")
    parser.add_argument("--w-worst", type=float, default=2.0,
                        help="Weight for Worst-Case")
    parser.add_argument("--w-best", type=float, default=0.5,
                        help="Weight for Best-Case")

    args = parser.parse_args()

    weights = {
        "mean": args.w_mean,
        "worst": args.w_worst,
        "best": args.w_best
    }

    console = Console()
    console.rule("[bold cyan]Calibration des seuils du critique PPO[/bold cyan]")
    console.print(
        f"Weights -> Mean: {args.w_mean}, Worst: {args.w_worst}, Best: {args.w_best}")
    console.print(f"Trials: {args.n_trials} | Routes per Trial: {args.runs}\n")

    plane_config = PlaneConfig(
        aircraft_type="A320", initial_lat=48.8566, initial_lon=2.3522)
    env_config = EnvironmentConfig()
    model_path = "ppo_flight_deviation_pretrained.zip"

    try:
        model = PPO.load(model_path, device="cpu")
        model.policy.set_training_mode(False)
        console.print(f"Loaded PPO model from {model_path}")
    except Exception as e:
        console.print(f"[bold red]Error loading model: {e}[/bold red]")
        return

    console.print(f"Generating {args.runs} benchmark routes...")
    generator = BenchmarkRouteGenerator(plane_config)
    routes_data = [generator.generate() for _ in range(args.runs)]

    console.print("Calculating FMS Baseline...", style="yellow")
    fms_fuel_rates = []

    for route, wind in track(routes_data, description="Running Baseline..."):
        env = OpenAPNavEnv(plane_config, env_config)
        env.set_nominal_route(route)
        env.set_wind_config(wind)
        stats = run_simulation(env, model=None)
        fms_fuel_rates.append(stats["fuel_per_nm"])

    avg_baseline = np.mean(fms_fuel_rates)
    console.print(f"Baseline Avg Fuel/NM: [bold]{avg_baseline:.4f}[/bold]\n")

    console.print(
        f"[bold green]Starting Optimization ({args.n_trials} trials)...[/bold green]")

    objective = OptimizerObjective(
        model, routes_data, fms_fuel_rates, plane_config, env_config, weights
    )

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler())

    with tqdm_optuna(total=args.n_trials, desc="Optimizing") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({"Best Score": f"{study.best_value:.2f}"})

        study.optimize(objective, n_trials=args.n_trials, callbacks=[callback])


    best_trial = study.best_trial

    console.print(
        f"\n[bold green]Meilleur essai (Score: {best_trial.value:.2f}):[/bold green]")
    console.print(
        f"  Threshold: [yellow]{best_trial.params['threshold']:.4f}[/yellow]")
    console.print(
        f"  Gamma:     [yellow]{best_trial.params['gamma']}[/yellow]")

    console.print("\n[bold]Métriques du meilleur essai:[/bold]")
    console.print(f"  Mean Gain:  {best_trial.user_attrs['mean_imp']:.2f}%")
    console.print(f"  Worst Case: {best_trial.user_attrs['worst_case']:.2f}%")
    console.print(f"  Best Case:  {best_trial.user_attrs['best_case']:.2f}%")

    # Show top 5 trials
    console.print("\n[bold]Top 5 Configurations:[/bold]")
    table = Table(box=box.ROUNDED)
    table.add_column("Rank", justify="center")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Threshold", justify="right", style="cyan")
    table.add_column("Gamma", justify="right")
    table.add_column("Mean %", justify="right")
    table.add_column("Worst %", justify="right")

    # Sort trials by value
    completed_trials = [t for t in study.trials if t.state ==
                        optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(
        completed_trials, key=lambda t: t.value, reverse=True)[:5]

    for rank, t in enumerate(top_trials, 1):
        mean_v = t.user_attrs.get('mean_imp', 0)
        worst_v = t.user_attrs.get('worst_case', 0)

        c_worst = "red" if worst_v < - \
            5 else ("yellow" if worst_v < 0 else "green")

        table.add_row(
            str(rank),
            f"{t.value:.2f}",
            f"{t.params['threshold']:.4f}",
            str(t.params['gamma']),
            f"{mean_v:.2f}%",
            f"[{c_worst}]{worst_v:.2f}%[/{c_worst}]"
        )

    console.print(table)

# Helper for Optuna progress bar


class tqdm_optuna:
    def __init__(self, total, desc="Optimizing"):
        from tqdm.rich import tqdm
        self.pbar = tqdm(total=total, desc=desc)

    def __enter__(self):
        return self.pbar

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.close()


if __name__ == "__main__":
    main()
