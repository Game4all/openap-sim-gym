"""
Script pour entraîner le modèle PPO de déviation avec tout le cursus d'apprentissage.
"""
import os
import gymnasium as gym
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from pydantic import BaseModel, Field
from rich import print as rprint, print_json
from xp_sim_gym.openap_env import OpenAPNavEnv, PlaneConfig, EnvironmentConfig
from xp_sim_gym import CurriculumPretrainingEnv


class CurriculumTrainingConfig(BaseModel):
    """
    Configuration du curriculum d'entraînement
    """
    aircraft_type: str = Field(
        description="Nom de l'engin avec lequel s'entraîner.")
    total_steps: int = Field(
        description="Durée totale de l'entraînement en étapes"
    )
    base_stage: int = Field(
        description="Phase du curriculum sur laquelle commencer l'entraînement")
    thresholds: dict[int, float] = Field(
        description="Un dict (stage -> seuil relatif, 0-1) pour configurer la durée de chaque phase du curriculum"
    )
    keep_env_config_prob: float = Field(
        default=0.1,
        description="Probabilité (0.0 à 1.0) de garder la même route et le même vent lors d'un reset."
    )

    @staticmethod
    def default_config():
        """Retourne la config par défaut"""
        return CurriculumTrainingConfig(
            total_steps=1_000_000,
            thresholds={
                1: 0.2,
                2: 0.4,
                3: 0.6,
                4: 0.8,
            },
            base_stage=1,
            aircraft_type="A320",
            keep_env_config_prob=0.0
        )


class CurriculumTrainingCallback(BaseCallback):
    """
    Callback d'entraînement pour changer le stage de pré-entraînement
    en fonction de l'avancement relatif du training.
    """

    def __init__(self, config: CurriculumTrainingConfig, verbose: int = 0):
        super().__init__(verbose)
        self.config = config
        self.stage = 1

        self.absolute_thresholds = {
            stage: int(rel * config.total_steps)
            for stage, rel in config.thresholds.items()
        }

        self.sorted_stages = sorted(self.absolute_thresholds.keys())

    def _on_step(self) -> bool:
        current_step = self.num_timesteps

        desired_stage = self.stage
        for stage in self.sorted_stages:
            if current_step >= self.absolute_thresholds[stage]:
                desired_stage = stage

        if desired_stage > self.stage:
            self.stage = desired_stage

            try:
                self.training_env.env_method(
                    "set_pretraining_stage", self.stage
                )
            except AttributeError:
                if hasattr(self.training_env, "set_pretraining_stage"):
                    self.training_env.set_pretraining_stage(self.stage)

        # logger la phase du curriculum d'entraînement pour aider a debug
        self.logger.record("curriculum/current_stage", self.stage)
        self.logger.record("curriculum/reuse_count",
                           self.training_env.get_attr("reuse_count")[0])

        # Log median reward if available
        if len(self.model.ep_info_buffer) > 0:
            rewards = [info["r"] for info in self.model.ep_info_buffer]
            median_reward = np.median(rewards)
            self.logger.record("rollout/ep_rew_median", median_reward)

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Script d'entraînement du modèle de déviation avec apprentissage par renforcement"
    )
    parser.add_argument("--total-timesteps", type=int,
                        help="Override du nombre d'étapes total pour l'entraînement")
    parser.add_argument("--n-steps", type=int,
                        help="Override du nombre d'étapes par rollout du PPO (PPO n_steps)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size pour l'entraînement")
    parser.add_argument(
        "--config",
        type=str,
        default="curriculum_config.json",
        help="Utilise la config de curriculum du fichier passé en paramètres pour l'entraînement"
    )
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            curriculum_config = CurriculumTrainingConfig.model_validate_json(
                f.read())
    else:
        rprint(
            f"[yellow]La configuration {args.config} n'existe pas. [bold]Création de la config par défaut.[/yellow][/bold]")
        curriculum_config = CurriculumTrainingConfig.default_config()
        with open(args.config, "w") as f:
            f.write(curriculum_config.model_dump_json(indent=4))
        return

    if args.total_timesteps:
        curriculum_config.total_steps = args.total_timesteps

    # Setup du dossier de log pour les run d'entraînement
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Setup de l'environement
    plane_config = PlaneConfig(aircraft_type=curriculum_config.aircraft_type)
    env_config = EnvironmentConfig()
    env = CurriculumPretrainingEnv(
        OpenAPNavEnv(plane_config, env_config),
        plane_config,
        prob_keep_config=curriculum_config.keep_env_config_prob,
        max_reuse_count=3,
        reuse_stages=[4, 5, 6]
    )

    # Initialization du modèle de PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=5e-4,
        ent_coef=0.02,
        n_steps=args.n_steps or 2048,
        batch_size=args.batch_size,
        gamma=0.99,
        device="cpu",
    )

    # Curriculum callback
    curr_callback = CurriculumTrainingCallback(
        config=curriculum_config,
        verbose=1,
    )

    rprint("[green]Début de l'entraînement avec la configuration suivante[/green]")
    print_json(curriculum_config.model_dump_json(indent=4))

    model.learn(
        total_timesteps=curriculum_config.total_steps,
        callback=curr_callback,
        progress_bar=True,
    )

    save_path = "ppo_flight_deviation_pretrained"
    model.save(save_path)
    rprint(
        f"[green]Fin de l'entraînement. [bold]Model sauvegardé dans {save_path}[/bold][/green]")


if __name__ == "__main__":
    main()
