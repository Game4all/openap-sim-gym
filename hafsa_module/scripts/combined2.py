"""
🏆 MODÈLE FINAL OPTIMISÉ : ppo_b737_fuel_optimizer_final
Basé sur les meilleurs résultats obtenus (5.8% économie)
Améliorations : Reward ajustée, hyperparamètres optimisés
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from openap import FuelFlow
from typing import Callable
import os

from xp_sim_gym.config import PlaneConfig, EnvironmentConfig
from xp_sim_gym.utils import GeoUtils
from xp_sim_gym.route_generator import RouteStageGenerator

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class B737FuelOptimizedEnv(gym.Env):
    """
    Environnement OPTIMISÉ pour économie de carburant
    Combine altitude/vitesse (Hafsa) + route (Lucas)
    """
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.ff_model = FuelFlow(ac="B738")
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Init proche des valeurs optimales
        self.alt = np.random.uniform(9000, 11000)  # Altitude croisière optimale
        self.speed = np.random.uniform(300, 360)   # Vitesse économique
        self.mass = np.random.uniform(60000, 75000)
        self.steps = 0
        self.prev_ff = 1.5
        self.prev_action = np.array([0.0, 0.0])
        self.alt_history = [self.alt]
        self.total_fuel_consumed = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.array([
            (self.alt - 10000) / 2500,
            (self.speed - 330) / 80,
            (self.prev_ff - 1.5) / 0.5,
            (self.mass - 67500) / 7500,
            self.prev_action[0],
            self.prev_action[1],
            self.total_fuel_consumed / 1000
        ], dtype=np.float32)
    
    def step(self, action):
        # Actions plus douces pour stabilité
        self.alt = np.clip(self.alt + (action[0] * 5), 2000, 12500)
        self.speed = np.clip(self.speed + (action[1] * 0.6), 200, 450)
        
        self.alt_history.append(self.alt)
        if len(self.alt_history) > 5:
            self.alt_history.pop(0)
        
        # Calcul fuel
        fuel_flow = self.ff_model.enroute(self.mass, self.speed, self.alt) * 2
        self.prev_ff = fuel_flow
        self.total_fuel_consumed += fuel_flow
        
        # REWARD OPTIMISÉE
        reward = 0.0
        
        # 1. Efficacité fuel (principal objectif)
        baseline_ff = 1.5  # Fuel flow baseline
        fuel_efficiency = baseline_ff - fuel_flow
        reward += fuel_efficiency * 20.0  # Récompense forte pour économie
        
        # 2. Bonus vitesse (maintenir performance)
        speed_bonus = (self.speed / 450) * 5.0
        reward += speed_bonus
        
        # 3. Stabilité vol (réduire oscillations)
        if len(self.alt_history) >= 5:
            alt_variance = np.std(self.alt_history)
            if alt_variance < 80:
                reward += 5.0
            else:
                reward -= alt_variance * 0.05
        
        # 4. Pénalité changements brusques
        action_change = np.abs(action - self.prev_action)
        if action_change[0] > 0.6: reward -= 8
        if action_change[1] > 0.6: reward -= 8
        
        # 5. Bonus actions douces
        if action_change[0] < 0.2 and action_change[1] < 0.2:
            reward += 3.0
        
        # 6. Pénalités limites
        if self.alt > 12000 or self.alt < 3000: reward -= 30
        if self.speed > 440 or self.speed < 220: reward -= 30
        
        # 7. Bonus altitude/vitesse optimales
        if 9500 < self.alt < 11000:  # Zone optimale
            reward += 2.0
        if 310 < self.speed < 370:   # Zone économique
            reward += 2.0

        self.mass -= fuel_flow * 1.0
        self.steps += 1
        self.prev_action = action.copy()
        
        info = {
            "altitude": self.alt,
            "speed_kts": self.speed,
            "fuel_flow": fuel_flow,
            "total_fuel": self.total_fuel_consumed,
            "efficiency": fuel_efficiency
        }
        
        return self._get_obs(), reward, self.steps >= 250, False, info

class OptimizedFlightEnv(gym.Env):
    """Environnement combiné OPTIMISÉ (route + fuel)"""
    def __init__(self, plane_config: PlaneConfig, env_config: EnvironmentConfig):
        super().__init__()
        
        self.route_generator = RouteStageGenerator(plane_config)
        self.plane_config = plane_config
        self.env_config = env_config
        
        self.flight_env = B737FuelOptimizedEnv()
        
        self.action_space = self.flight_env.action_space
        
        obs_dim = 7 + 4
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.route, self.wind_streams = self.route_generator.generate(stage=4)
        self.current_waypoint_idx = 1
        
        start_wp = self.route[0]
        self.lat = start_wp['lat']
        self.lon = start_wp['lon']
        
        obs_flight, _ = self.flight_env.reset(seed=seed)
        
        route_obs = self._get_route_obs()
        obs = np.concatenate([obs_flight, route_obs])
        
        self.total_distance_flown = 0
        self.route_distance = self._calculate_total_route_distance()
        
        return obs, {}
    
    def step(self, action):
        obs_flight, reward_flight, done_flight, trunc_flight, info_flight = self.flight_env.step(action)
        
        self._update_position(info_flight)
        route_obs = self._get_route_obs()
        
        xte = self._calculate_xte()
        
        # Reward route optimisée
        if abs(xte) < 8:
            reward_route = 3.0
        elif abs(xte) < 20:
            reward_route = 1.0
        else:
            reward_route = -abs(xte) * 0.15
        
        reward = reward_flight + reward_route
        
        done = done_flight or self.current_waypoint_idx >= len(self.route)
        
        obs = np.concatenate([obs_flight, route_obs])
        
        info = {
            **info_flight,
            "xte": xte,
            "route_progress": self.current_waypoint_idx / len(self.route),
            "route_distance": self.route_distance,
            "distance_flown": self.total_distance_flown
        }
        
        return obs, reward, done, trunc_flight, info
    
    def _get_route_obs(self):
        if self.current_waypoint_idx < len(self.route):
            target_wp = self.route[self.current_waypoint_idx]
            dist = GeoUtils.haversine_dist(
                self.lat, self.lon, target_wp['lat'], target_wp['lon']
            )
            bearing = GeoUtils.bearing(
                self.lat, self.lon, target_wp['lat'], target_wp['lon']
            )
            xte = self._calculate_xte()
            progress = self.current_waypoint_idx / len(self.route)
        else:
            dist = 0
            bearing = 0
            xte = 0
            progress = 1.0
        
        return np.array([
            dist / 1000,
            bearing / 360,
            xte / 50,
            progress
        ], dtype=np.float32)
    
    def _calculate_xte(self):
        if self.current_waypoint_idx >= len(self.route) or self.current_waypoint_idx == 0:
            return 0.0
        
        prev_wp = self.route[self.current_waypoint_idx - 1]
        target_wp = self.route[self.current_waypoint_idx]
        
        return GeoUtils.cross_track_error(
            self.lat, self.lon,
            prev_wp['lat'], prev_wp['lon'],
            target_wp['lat'], target_wp['lon']
        )
    
    def _update_position(self, info):
        speed_ms = info['speed_kts'] * 0.5144
        distance_m = speed_ms * 60
        self.total_distance_flown += distance_m / 1852
        
        if self.current_waypoint_idx < len(self.route):
            target_wp = self.route[self.current_waypoint_idx]
            dist = GeoUtils.haversine_dist(
                self.lat, self.lon, target_wp['lat'], target_wp['lon']
            )
            
            if dist < 15:
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx < len(self.route):
                    self.lat = target_wp['lat']
                    self.lon = target_wp['lon']
    
    def _calculate_total_route_distance(self):
        total = 0
        for i in range(len(self.route) - 1):
            wp1, wp2 = self.route[i], self.route[i+1]
            total += GeoUtils.haversine_dist(
                wp1['lat'], wp1['lon'], wp2['lat'], wp2['lon']
            )
        return total

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'altitude' in info:
                self.logger.record("env/altitude", info['altitude'])
            if 'fuel_flow' in info:
                self.logger.record("env/fuel_flow", info['fuel_flow'])
            if 'speed_kts' in info:
                self.logger.record("env/speed", info['speed_kts'])
            if 'xte' in info:
                self.logger.record("env/xte", info['xte'])
            if 'efficiency' in info:
                self.logger.record("env/efficiency", info['efficiency'])
        return True

# ==================== ENTRAÎNEMENT ====================

if __name__ == "__main__":
    print("="*70)
    print("🏆 ENTRAÎNEMENT MODÈLE FINAL OPTIMISÉ")
    print("="*70)
    
    log_dir = "./logs_fuel_optimizer_final/"
    os.makedirs(log_dir, exist_ok=True)
    
    plane_config = PlaneConfig(aircraft_type="B738")
    env_config = EnvironmentConfig()
    
    env = OptimizedFlightEnv(plane_config, env_config)
    
    # Hyperparamètres OPTIMISÉS
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(2e-4),
        n_steps=4096,        # Plus de steps pour meilleure exploration
        batch_size=256,
        n_epochs=15,         # Plus d'epochs pour meilleur apprentissage
        ent_coef=0.008,      # Encourage exploration
        gamma=0.999,         # Horizon long terme
        clip_range=0.18,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    print("\n📊 TensorBoard : tensorboard --logdir ./logs_fuel_optimizer_final/")
    print("🎯 Objectif : Dépasser 5.8% économie")
    print("⏱️  Durée : 700 000 steps (~45 min)")
    print("="*70 + "\n")
    
    callback = TensorboardCallback()
    
    model.learn(
        total_timesteps=700_000,
        callback=callback,
        progress_bar=True
    )
    
    model.save("ppo_b737_fuel_optimizer_final")
    
    print("\n" + "="*70)
    print("✅ Modèle sauvegardé : ppo_b737_fuel_optimizer_final.zip")
    print("="*70)