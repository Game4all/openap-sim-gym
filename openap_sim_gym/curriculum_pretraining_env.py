import gymnasium as gym
from typing import Optional
import random
from .openap_env import OpenAPNavEnv
from .route_generator import RouteStageGenerator
from .config import PlaneConfig
import tensorboard


class CurriculumPretrainingEnv(gym.Wrapper):
    """
    An environment that wraps another environment.
    This environment contains the logic for pre-training with a learning curriculum designed to help the agent start learning effectively.
    """

    def __init__(self, env: OpenAPNavEnv, plane_config: PlaneConfig, seed: Optional[int] = None, prob_keep_config: float = 0.0, max_reuse_count: int = 2, reuse_stages: list[int] = [4, 5, 6]):
        super().__init__(env)
        self.env = env  # type: OpenAPNavEnv
        self.route_generator = RouteStageGenerator(plane_config, seed=seed)
        self.stage = 1
        self.prob_keep_config = prob_keep_config

        self.last_route = None
        self.last_wind_streams = None
        self.reuse_count = 0
        self.max_reuse_count = max_reuse_count
        self.reuse_stages = reuse_stages

    def set_pretraining_stage(self, stage: int):
        if stage not in [1, 2, 3, 4, 5, 6]:
            raise ValueError(f"Invalid stage {stage}")

        self.stage = stage
        print(f"[INFO] Curriculum Wrapper: Switched to Stage {stage}")

    def set_keep_config_prob(self, prob: float):
        self.prob_keep_config = prob

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        keep_prev_config = False
        if self.last_route is not None and random.random() < self.prob_keep_config and self.reuse_count < self.max_reuse_count and self.stage in self.reuse_stages:
            keep_prev_config = True

        if keep_prev_config:
            route = self.last_route
            wind_streams = self.last_wind_streams
            self.reuse_count += 1
        else:
            route = self.route_generator.generate_route(self.stage)

            start_lat = route[0].get(
                'lat', self.env.plane_config.initial_lat) if route else self.env.plane_config.initial_lat
            start_lon = route[0].get(
                'lon', self.env.plane_config.initial_lon) if route else self.env.plane_config.initial_lon

            wind_streams = self.route_generator.generate_wind(
                self.stage,
                route,
                start_lat,
                start_lon
            )

            self.last_route = route
            self.last_wind_streams = wind_streams
            self.reuse_count = 0

        self.env.set_nominal_route(route)
        self.env.set_wind_config(wind_streams)

        return self.env.reset(seed=seed, options=options)
