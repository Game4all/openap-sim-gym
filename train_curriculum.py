"""
Script pour entraîner le modèle PPO de déviation avec tout le cursus d'apprentissage.
"""
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from xp_sim_gym.openap_env import OpenAPNavEnv, PlaneEnvironmentConfig
import argparse


class CurriculumTrainingCallback(BaseCallback):
    """
    Callback d'entraînement pour changer le stage de pré-entraînement en fonction de l'avancement global.
    """

    def __init__(self, verbose=0):
        super(CurriculumTrainingCallback, self).__init__(verbose)
        self.stage = 1

        # Define stage transitions (timestep thresholds)
        self.thresholds = {
            1: 200_000,
            2: 400_000,
            3: 600_000,
            4: 800_000,
        }

    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        desired_stage = self.stage

        if self.stage == 1 and current_step > self.thresholds[1]:
            desired_stage = 2
        elif self.stage == 2 and current_step > self.thresholds[2]:
            desired_stage = 3
        elif self.stage == 3 and current_step > self.thresholds[3]:
            desired_stage = 4

        if desired_stage > self.stage:
            self.stage = desired_stage

            try:
                self.training_env.env_method(
                    "set_pretraining_stage", self.stage
                )
            except AttributeError:
                if hasattr(self.training_env, "set_pretraining_stage"):
                    self.training_env.set_pretraining_stage(self.stage)

            if self.verbose > 0:
                print(
                    f"\n[Curriculum] Advancing to Stage {self.stage} at step {current_step}"
                )

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Script d'entraînement du modèle de déviation avec apprentissage par renforcement"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total number of training timesteps",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps per rollout (PPO n_steps)",
    )
    parser.add_argument("--batch-size",
                        type=int, default=64)

    args = parser.parse_args()

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    config = PlaneEnvironmentConfig(aircraft_type="A320")
    env = OpenAPNavEnv(config=config)

    # Initialize Agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        ent_coef=0.02,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        device="cpu",
    )

    curr_callback = CurriculumTrainingCallback(verbose=1)

    print("Début de l'entraînement")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=curr_callback,
        progress_bar=True,
    )

    save_path = "ppo_flight_deviation_pretrained"
    model.save(save_path)
    print(f"Fin de l'entraînement. Model sauvegardé dans {save_path}")


if __name__ == "__main__":
    main()
