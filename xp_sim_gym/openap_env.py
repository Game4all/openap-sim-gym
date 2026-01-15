import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from .utils import GeoUtils
from openap import FuelFlow
from .constants import (
    MAX_DEVIATION_SEGMENTS, MAX_XTRACK_ERROR_NM,
    MAX_ALT, MAX_SPD, MAX_FUEL, MAX_WIND, MAX_DIST
)

class OpenAPNavEnv(gym.Env):
    """
    Un environement gymnasium utilisant OpenAP pour simuler un avion. 
    Cet environement vise à permettre le pré-entraînement d'un agent pour optimiser les déviations de vol.

    **Observations:**
    Vecteur numpy avec `6 + 4 + (4 * lookahead_count)` éléments :
    
    1.  **État de l'Avion (6)** :
        *   `norm_alt` : Altitude actuelle / MAX_ALT (45,000 ft)
        *   `norm_tas` : Vitesse Vraie (TAS) / MAX_SPD (600 kts)
        *   `norm_gs` : Vitesse Sol (GS) / MAX_SPD (600 kts)
        *   `norm_fuel` : Quantité de Carburant / MAX_FUEL (20,000 kg)
        *   `norm_wu` : Composante Vent U / MAX_WIND (200 kts)
        *   `norm_wv` : Composante Vent V / MAX_WIND (200 kts)
    
    2.  **Contexte de la Route (4)** :
        *   `xte` : Erreur Latérale (Cross-Track Error) / MAX_XTE (50 NM)
        *   `dist_to_wpt` : Distance au prochain waypoint / MAX_DIST (1000 NM)
        *   `brg_err` : Erreur de Cap vers la route / 180 degrés
        *   `dist_to_dest` : Distance à la destination / (2 * MAX_DIST)

    3.  **Anticipation (Lookahead) (4 * N)** : 
        Pour chacun des `N` (par défaut 3) prochains waypoints :
        *   `d` : Distance depuis l'avion / MAX_DIST
        *   `rel_b` : Relèvement relatif / 180 degrés
        *   `along` : Composante de vent longitudinale / MAX_WIND
        *   `cross` : Composante de vent latérale / MAX_WIND

    **Actions:**
    Un vecteur de 2 éléments :
    1.  `Heading Offset` : Déviation en degrés relative au cap actuel. Plage : [-30.0, 30.0].
    2.  `Duration` : Temps de maintien de la déviation en minutes. Plage : [5.0, 20.0].
    """
    metadata = {"render_modes": [], "render_fps": 1}

    def __init__(self, aircraft_type='A320', lookahead_count=3, nominal_route=None):
        super().__init__()
        
        self.aircraft_type = aircraft_type
        self.fuel_flow_model = FuelFlow(ac=aircraft_type)
        
        self.lookahead_count = lookahead_count
        self.nominal_route = nominal_route if nominal_route else []
        self.current_waypoint_idx = 0
        self.stage = 1 # Default to stage 1 (Easy)
        
        obs_dim = 6 + 4 + (4 * self.lookahead_count)
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([-30.0, 5.0]), 
            high=np.array([30.0, 20.0]), 
            shape=(2,), 
            dtype=np.float32
        )
        
        # Initial State placeholders
        self.lat = 0.0
        self.lon = 0.0
        self.alt_m = 10000.0 # ~33k ft default
        self.tas_ms = 230.0 # ~450 kts
        self.gs_ms = 230.0
        self.heading_mag = 0.0
        self.current_fuel_kg = 15000.0
        
        self.wind_u = 0.0
        self.wind_v = 0.0
        
        self.steps_taken = 0
        
    def set_pretraining_stage(self, stage):
        """Règle le niveau de difficulté de l'environnement pour le pré-entraînement"""
        if stage not in [1, 2, 3]:
            print(f"Warning: Etape {stage} invalide.")
            return

        self.stage = stage
        print(f"OpenAPNavEnv switched to Stage {stage}")
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        self.current_waypoint_idx = 0
        
        if not self.nominal_route or self.stage > 0:
             self.nominal_route = self._generate_route_for_stage()

        # Initialize state based on first waypoint
        if self.nominal_route:
            start_node = self.nominal_route[0]
            self.lat = start_node.get('lat', 0.0)
            self.lon = start_node.get('lon', 0.0)
            self.alt_m = start_node.get('alt', 10058.4) 
            
            if len(self.nominal_route) > 1:
                next_node = self.nominal_route[1]
                self.heading_mag = GeoUtils.bearing(self.lat, self.lon, next_node['lat'], next_node['lon'])
            else:
                self.heading_mag = 0.0
        
        self.current_fuel_kg = 15000.0 
        self.tas_ms = 230.0 
        self.gs_ms = self.tas_ms 
        
        # Wind initialization based on Stage
        self._init_wind()
        
        return self._get_observation(), {}

    def step(self, action):
        heading_offset_deg = float(action[0])
        duration_min = float(action[1])
        duration_sec = duration_min * 60.0
        
        # 1. Apply Action (Heading)
        target_heading = (self.heading_mag + heading_offset_deg) % 360.0
        
        # 2. Kinematics Simulation
        # Assume constant wind for the segment
        # Update Ground Speed vector
        heading_rad = math.radians(target_heading)
        
        # TAS vector
        tas_n = self.tas_ms * math.cos(heading_rad)
        tas_e = self.tas_ms * math.sin(heading_rad)
        
        # Ground Speed vector
        gs_n = tas_n + self.wind_v # wind_v is North component? XPlane uses U=East, V=North usually? 
        # In env.py: 
        # w_rad = math.radians(w_dir_deg)
        # self.wind_u = -w_spd_kt * math.sin(w_rad)  (East?)
        # self.wind_v = -w_spd_kt * math.cos(w_rad)  (North?)
        # Let's assume standard u=East, v=North.
        # Note: In env.py, wind_u and wind_v are derived from wind direction (blowing FROM).
        # We need wind vector (blowing TO) for ground speed calc. 
        # If env.py wind_u/v are "blowing to" components, we add them. 
        # env.py: -w_spd * sin(dir) -> suggests converting "from" to "uv components". 
        # typically wind direction is "from".
        # let's assume wind_u, wind_v are the flow vector components acting on the plane.
        
        gs_e = tas_e + self.wind_u
        gs_n = tas_n + self.wind_v
        
        self.gs_ms = math.sqrt(gs_e**2 + gs_n**2)
        track_rad = math.atan2(gs_e, gs_n)
        track_deg = (math.degrees(track_rad) + 360) % 360
        
        # Distance flown in duration
        dist_m = self.gs_ms * duration_sec
        dist_nm = dist_m / 1852.0
        
        # Update Position
        # Simple flat earth approximation for short distances or use geoutils if available
        # Using a simplified lat/lon update based on bearing and distance
        # New Lat/Lon
        # d_lat = dist_nm * cos(track) / 60
        # d_lon = dist_nm * sin(track) / (60 * cos(lat))
        
        delta_lat = (dist_nm * math.cos(track_rad)) / 60.0
        self.lat += delta_lat
        
        delta_lon = (dist_nm * math.sin(track_rad)) / (60.0 * math.cos(math.radians(self.lat)))
        self.lon += delta_lon
        
        # Update Heading (Simulating the AP holding the heading)
        self.heading_mag = target_heading 
        
        # 3. Fuel Consumption Simulation (OpenAP)
        # enroute(mass, tas, alt, path_angle) -> fuel flow in kg/s
        ff_kg_s = self.fuel_flow_model.enroute(
            mass=self.current_fuel_kg + 40000, # Approximate ZFW + Fuel. OpenAP needs Total Mass.
            tas=self.tas_ms / 0.514444, # to kts
            alt=self.alt_m / 0.3048, # to ft
        )
        
        fuel_consumed = ff_kg_s * duration_sec
        self.current_fuel_kg -= fuel_consumed
        
        # 4. Check Waypoint/Segment Logic
        self._check_waypoint_progression()
        
        # 5. Reward Calculation
        reward = -fuel_consumed
        reward -= 0.1 * duration_min
        xte_nm = self._calculate_xte()
        reward -= 0.5 * abs(xte_nm)
        
        terminated = False
        truncated = False
        
        if self.steps_taken >= MAX_DEVIATION_SEGMENTS:
            truncated = True
        
        if abs(xte_nm) > MAX_XTRACK_ERROR_NM:
            reward -= 100.0
            terminated = True
            
        self.steps_taken += 1
        
        return self._get_observation(), reward, terminated, truncated, {}

    def _check_waypoint_progression(self):
        if self.current_waypoint_idx < len(self.nominal_route):
            target_wp = self.nominal_route[self.current_waypoint_idx]
            dist = GeoUtils.haversine_dist(self.lat, self.lon, target_wp['lat'], target_wp['lon'])
            
            # Simple sequencing logic: if we pass abeam or get close
            # For now, just proximity
            if dist < 5.0: 
                 self.current_waypoint_idx += 1

    def _calculate_xte(self):
        if self.current_waypoint_idx >= len(self.nominal_route):
             return 0.0
             
        target_wp = self.nominal_route[self.current_waypoint_idx]
        
        if self.current_waypoint_idx == 0:
            return 0.0
        
        prev_wp = self.nominal_route[self.current_waypoint_idx - 1]
        
        return GeoUtils.cross_track_error(
            self.lat, self.lon, 
            prev_wp['lat'], prev_wp['lon'], 
            target_wp['lat'], target_wp['lon']
        )
        
    def _sample_wind_at(self, lat, lon, alt):
        return self.wind_u, self.wind_v

    def _init_wind(self):
        """Initialize wind based on current stage."""
        if self.stage == 1:
            # Stage 1: Zero wind
            self.wind_u = 0.0
            self.wind_v = 0.0
        elif self.stage == 2:
            # Stage 2: Constant moderate wind (random direction)
            spd = np.random.uniform(10, 30) # kts
            bng = np.random.uniform(0, 360)
            
            # Convert to U/V (assuming MET convention 'wind from')
            # But we want flow components for GS calc.
            # u = -spd * sin(rad)
            # v = -spd * cos(rad)
            rad = math.radians(bng)
            self.wind_u = -spd * math.sin(rad) * 0.514444 # to m/s
            self.wind_v = -spd * math.cos(rad) * 0.514444
            
        elif self.stage == 3:
            # Stage 3: Stronger/Variable wind (randomized per episode)
            spd = np.random.uniform(30, 80) # Stronger
            bng = np.random.uniform(0, 360)
            rad = math.radians(bng)
            self.wind_u = -spd * math.sin(rad) * 0.514444
            self.wind_v = -spd * math.cos(rad) * 0.514444
            
    def _generate_route_for_stage(self):
        """Generates a random route based on difficulty stage."""
        # Base location (Paris roughly)
        lat_0, lon_0 = 48.0, 2.0
        
        route = []
        route.append({'lat': lat_0, 'lon': lon_0, 'alt': 33000})
        
        if self.stage == 1:
            # Short straight route: 1 segment, ~50-100 NM
            dist_nm = np.random.uniform(50, 100)
            bearing = np.random.uniform(0, 360)
            
            # Rough calc for next point
            # 1 deg lat ~ 60 NM
            d_lat = dist_nm / 60.0 * math.cos(math.radians(bearing))
            d_lon = dist_nm / 60.0 * math.sin(math.radians(bearing)) / math.cos(math.radians(lat_0))
            
            route.append({'lat': lat_0 + d_lat, 'lon': lon_0 + d_lon, 'alt': 33000})
            
        elif self.stage == 2:
            # Medium route: 2-3 segments, ~200 NM total
            num_segments = np.random.randint(2, 4)
            current_lat, current_lon = lat_0, lon_0
            current_bng = np.random.uniform(0, 360)
            
            for _ in range(num_segments):
                dist_nm = np.random.uniform(50, 80)
                # Small turn
                turn = np.random.uniform(-30, 30)
                current_bng = (current_bng + turn) % 360
                
                d_lat = dist_nm / 60.0 * math.cos(math.radians(current_bng))
                d_lon = dist_nm / 60.0 * math.sin(math.radians(current_bng)) / math.cos(math.radians(current_lat))
                
                current_lat += d_lat
                current_lon += d_lon
                route.append({'lat': current_lat, 'lon': current_lon, 'alt': 33000})
                
        elif self.stage == 3:
             # Long/Complex route: 4-6 segments, ~400-600 NM
            num_segments = np.random.randint(4, 7)
            current_lat, current_lon = lat_0, lon_0
            current_bng = np.random.uniform(0, 360)
            
            for _ in range(num_segments):
                dist_nm = np.random.uniform(60, 100)
                turn = np.random.uniform(-60, 60) # Sharper turns
                current_bng = (current_bng + turn) % 360
                
                d_lat = dist_nm / 60.0 * math.cos(math.radians(current_bng))
                d_lon = dist_nm / 60.0 * math.sin(math.radians(current_bng)) / math.cos(math.radians(current_lat))
                
                current_lat += d_lat
                current_lon += d_lon
                route.append({'lat': current_lat, 'lon': current_lon, 'alt': 33000})
                
        return route

    def _get_observation(self):
        # reuse logic from env.py but adapted for local vars
        norm_alt = self.alt_m / (MAX_ALT * 0.3048) 
        norm_tas = self.tas_ms / (MAX_SPD * 0.514444)
        norm_gs  = self.gs_ms / (MAX_SPD * 0.514444)
        norm_fuel= self.current_fuel_kg / MAX_FUEL
        norm_wu = self.wind_u / (MAX_WIND * 0.514444)
        norm_wv = self.wind_v / (MAX_WIND * 0.514444)
        
        state_obs = np.array([norm_alt, norm_tas, norm_gs, norm_fuel, norm_wu, norm_wv], dtype=np.float32)
        
        # Route Obs
        if self.current_waypoint_idx < len(self.nominal_route):
            target_wp = self.nominal_route[self.current_waypoint_idx]
            if self.current_waypoint_idx == 0:
                # If start, treat start pos as prev
                # We need the route start, not current pos if we want stable XTE
                # But for simplicity, let's assume route[0] is start point and we fly to route[1]...
                # Actually usually route is sequence of waypoints TO fly to.
                # Let's assume we start at route[0] and fly to route[1].
                # If index=0, we are flying to route[0].
                prev_lat, prev_lon = self.lat, self.lon # Ephemeral prev
            else:
                prev_wp = self.nominal_route[self.current_waypoint_idx - 1]
                prev_lat, prev_lon = prev_wp['lat'], prev_wp['lon']
            
            target_lat, target_lon = target_wp['lat'], target_wp['lon']
            
            xte = GeoUtils.cross_track_error(self.lat, self.lon, prev_lat, prev_lon, target_lat, target_lon)
            dist_to_wpt = GeoUtils.haversine_dist(self.lat, self.lon, target_lat, target_lon)
            
            desired_track = GeoUtils.bearing(self.lat, self.lon, target_lat, target_lon)
            brg_err = (desired_track - self.heading_mag + 180) % 360 - 180
            
            last_wp = self.nominal_route[-1]
            dist_to_dest = GeoUtils.haversine_dist(self.lat, self.lon, last_wp['lat'], last_wp['lon'])
            
        else:
            xte = 0.0
            dist_to_wpt = 0.0
            brg_err = 0.0
            dist_to_dest = 0.0

        route_obs = np.array([
            xte / MAX_XTRACK_ERROR_NM,
            dist_to_wpt / MAX_DIST,
            brg_err / 180.0,
            dist_to_dest / (MAX_DIST * 2)
        ], dtype=np.float32)
        
        lookahead_obs = []
        for i in range(self.lookahead_count):
            idx = self.current_waypoint_idx + i
            if idx < len(self.nominal_route):
                wp = self.nominal_route[idx]
                d = GeoUtils.haversine_dist(self.lat, self.lon, wp['lat'], wp['lon'])
                b = GeoUtils.bearing(self.lat, self.lon, wp['lat'], wp['lon'])
                rel_b = (b - self.heading_mag + 180) % 360 - 180
                
                w_u, w_v = self._sample_wind_at(wp['lat'], wp['lon'], wp.get('alt', 33000))
                
                if idx == 0:
                     leg_brg = GeoUtils.bearing(self.lat, self.lon, wp['lat'], wp['lon'])
                else:
                     p_wp = self.nominal_route[idx-1]
                     leg_brg = GeoUtils.bearing(p_wp['lat'], p_wp['lon'], wp['lat'], wp['lon'])
                
                leg_rad = math.radians(leg_brg)
                along = w_u * math.sin(leg_rad) + w_v * math.cos(leg_rad)
                cross = w_u * math.cos(leg_rad) - w_v * math.sin(leg_rad)
                
                lookahead_obs.extend([
                    d / MAX_DIST,
                    rel_b / 180.0,
                    along / (MAX_WIND * 0.5144),
                    cross / (MAX_WIND * 0.5144)
                ])
            else:
                lookahead_obs.extend([1.0, 0.0, 0.0, 0.0])
                
        return np.concatenate([state_obs, route_obs, np.array(lookahead_obs, dtype=np.float32)])
