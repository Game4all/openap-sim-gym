import gymnasium as gym
from typing import Optional
from .openap_env import OpenAPNavEnv
from .route_generator import RouteStageGenerator
from .config import PlaneConfig
import tensorboard

class CurriculumPretrainingEnv(gym.Wrapper):
    """
    Un environnement gymnasium qui entourne un autre environnement gymnasium.
    Cet environnement contient la logique de pré-entraînement avec un curriculum d'apprentissage qui vise à permettre à l'agent de bien commencer son apprentissage.
    """
    def __init__(self, env: OpenAPNavEnv, plane_config: PlaneConfig, seed: Optional[int] = None):
        super().__init__(env)
        self.env = env # type: OpenAPNavEnv
        self.route_generator = RouteStageGenerator(plane_config, seed=seed)
        self.stage = 1
        
    def set_pretraining_stage(self, stage: int):
        """Sets the difficulty stage for route/wind generation."""
        if stage not in [1, 2, 3, 4, 5, 6]:
            raise ValueError(f"Invalid stage {stage}")
        
        self.stage = stage
        print(f"[INFO] Curriculum Wrapper: Switched to Stage {stage}")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Generate new episode configuration
        route = self.route_generator.generate_route(self.stage)
        
        # Determine start params
        start_lat = route[0].get('lat', self.env.plane_config.initial_lat) if route else self.env.plane_config.initial_lat
        start_lon = route[0].get('lon', self.env.plane_config.initial_lon) if route else self.env.plane_config.initial_lon

        wind_streams = self.route_generator.generate_wind(
            self.stage, 
            route, 
            start_lat, 
            start_lon
        )
        
        # Inject into the environment
        self.env.set_nominal_route(route)
        self.env.set_wind_config(wind_streams)
        
        return self.env.reset(seed=seed, options=options)
