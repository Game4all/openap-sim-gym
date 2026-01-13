import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import time
from .utils import GeoUtils

class XPlaneDevEnv(gym.Env):
    """
    Un environnement Gymnasium pour optimiser les déviations de vol dans X-Plane afin d'exploiter les champs de vent et autres perturbations météo intéressantes

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

    Une fois la durée écoulée, l'environnement tente automatiquement de rejoindre la route nominale.
    """
    metadata = {"render_modes": [], "render_fps": 1}

    def __init__(self, xpc_client, lookahead_count=3, nominal_route=None):
        super(XPlaneDevEnv, self).__init__()
        
        self.xpc = xpc_client
        self.lookahead_count = lookahead_count
        self.nominal_route = nominal_route if nominal_route else []
        self.current_waypoint_idx = 0
        
        # Constantes
        self.DT_DECISION = 300
        self.MAX_DEVIATION_SEGMENTS = 20
        self.MAX_XTRACK_ERROR_NM = 50.0
        
        # Valeurs max pour normalisation
        self.MAX_ALT = 45000.0
        self.MAX_SPD = 600.0 
        self.MAX_FUEL = 20000.0 
        self.MAX_WIND = 200.0 
        self.MAX_DIST = 1000.0 
        
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
        
        self.steps_taken = 0
        self.current_fuel_kg = 0.0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        self.current_waypoint_idx = 0
        self._engage_autopilot()
        self._update_state()
        return self._get_observation(), {}
        
    def _engage_autopilot(self):
        if self.xpc:
            # Essayer d'activer l'AP en mode maintien de cap (Heading Hold)
            self.xpc.sendDREF("sim/cockpit/autopilot/autopilot_mode", 2)
            self.xpc.sendDREF("sim/cockpit/autopilot/heading_mode", 1)  

    def step(self, action):
        heading_offset_deg = float(action[0])
        duration_min = float(action[1])
        duration_sec = duration_min * 60.0
        
        current_heading = self.heading_mag
        target_heading = (current_heading + heading_offset_deg) % 360.0
        
        self._send_heading_command(target_heading)
        self._wait_sim_time(duration_sec)
        self._navigate_direct_to_next_waypoint()
        
        fuel_start = self.current_fuel_kg
        self._update_state()
        fuel_end = self.current_fuel_kg
        fuel_consumed = fuel_start - fuel_end
        
        reward = -fuel_consumed
        reward -= 0.1 * duration_min
        xte_nm = self._calculate_xte()
        reward -= 0.5 * abs(xte_nm)
        
        terminated = False
        truncated = False
        
        if self.steps_taken >= self.MAX_DEVIATION_SEGMENTS:
            truncated = True
        
        if abs(xte_nm) > self.MAX_XTRACK_ERROR_NM:
            reward -= 100.0
            terminated = True
            
        self.steps_taken += 1
        
        return self._get_observation(), reward, terminated, truncated, {}

    def _update_state(self):
        if not self.xpc:
            return

        drefs = [
            "sim/flightmodel/position/latitude",
            "sim/flightmodel/position/longitude",
            "sim/flightmodel/position/elevation",
            "sim/flightmodel/position/true_airspeed",
            "sim/flightmodel/position/groundspeed",
            "sim/flightmodel/position/mag_psi",
            "sim/flightmodel/weight/m_fuel_total",
            "sim/weather/wind_speed_kt",
            "sim/weather/wind_direction_degt"
        ]
        
        values = self.xpc.getDREFs(drefs)
        
        self.lat = values[0][0]
        self.lon = values[1][0]
        self.alt_m = values[2][0]
        self.tas_ms = values[3][0]
        self.gs_ms = values[4][0]
        self.heading_mag = values[5][0]
        self.current_fuel_kg = values[6][0]
        
        w_spd_kt = values[7][0]
        w_dir_deg = values[8][0]
        w_rad = math.radians(w_dir_deg)
        self.wind_u = -w_spd_kt * math.sin(w_rad) 
        self.wind_v = -w_spd_kt * math.cos(w_rad)

    def _get_observation(self):
        norm_alt = self.alt_m / (self.MAX_ALT * 0.3048) # m to ft conversion if necessary, or standardize
        norm_tas = self.tas_ms / (self.MAX_SPD * 0.514444)
        norm_gs  = self.gs_ms / (self.MAX_SPD * 0.514444)
        norm_fuel= self.current_fuel_kg / self.MAX_FUEL
        norm_wu = self.wind_u / (self.MAX_WIND * 0.514444)
        norm_wv = self.wind_v / (self.MAX_WIND * 0.514444)
        
        state_obs = np.array([norm_alt, norm_tas, norm_gs, norm_fuel, norm_wu, norm_wv], dtype=np.float32)
        
        if self.current_waypoint_idx < len(self.nominal_route):
            target_wp = self.nominal_route[self.current_waypoint_idx]
            # For XTE calculation, we need the previous point on the nominal route
            if self.current_waypoint_idx == 0:
                # If at the first waypoint, use current aircraft position as 'previous'
                prev_lat, prev_lon = self.lat, self.lon
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
            # If all waypoints are passed, set route observations to zero
            xte = 0.0
            dist_to_wpt = 0.0
            brg_err = 0.0
            dist_to_dest = 0.0

        route_obs = np.array([
            xte / self.MAX_XTRACK_ERROR_NM,
            dist_to_wpt / self.MAX_DIST,
            brg_err / 180.0,
            dist_to_dest / (self.MAX_DIST * 2)
        ], dtype=np.float32)
        
        lookahead_obs = []
        for i in range(self.lookahead_count):
            idx = self.current_waypoint_idx + i
            if idx < len(self.nominal_route):
                wp = self.nominal_route[idx]
                d = GeoUtils.haversine_dist(self.lat, self.lon, wp['lat'], wp['lon'])
                b = GeoUtils.bearing(self.lat, self.lon, wp['lat'], wp['lon'])
                rel_b = (b - self.heading_mag + 180) % 360 - 180
                
                # Sample wind at waypoint location (simplified for now)
                w_u, w_v = self._sample_wind_at(wp['lat'], wp['lon'], wp['alt'])
                
                # Calculate leg bearing for wind components
                if idx == 0:
                     leg_brg = GeoUtils.bearing(self.lat, self.lon, wp['lat'], wp['lon'])
                else:
                     p_wp = self.nominal_route[idx-1]
                     leg_brg = GeoUtils.bearing(p_wp['lat'], p_wp['lon'], wp['lat'], wp['lon'])
                
                leg_rad = math.radians(leg_brg)
                # Project wind onto along-track and cross-track components
                along = w_u * math.sin(leg_rad) + w_v * math.cos(leg_rad)
                cross = w_u * math.cos(leg_rad) - w_v * math.sin(leg_rad)
                
                lookahead_obs.extend([
                    d / self.MAX_DIST,
                    rel_b / 180.0,
                    along / (self.MAX_WIND * 0.5144), # Normalize wind components
                    cross / (self.MAX_WIND * 0.5144)
                ])
            else:
                # If no more waypoints, pad with default values (e.g., 1.0 for distance, 0.0 for others)
                lookahead_obs.extend([1.0, 0.0, 0.0, 0.0])
                
        return np.concatenate([state_obs, route_obs, np.array(lookahead_obs, dtype=np.float32)])

    def _send_heading_command(self, heading_deg):
        """Sends a heading command to the X-Plane autopilot."""
        if self.xpc:
            self.xpc.sendDREF("sim/cockpit/autopilot/heading_mag", heading_deg)

    def _navigate_direct_to_next_waypoint(self):
        """Commands the AP to fly direct to the active waypoint."""
        if self.current_waypoint_idx < len(self.nominal_route):
            target_wp = self.nominal_route[self.current_waypoint_idx]
            bearing = GeoUtils.bearing(self.lat, self.lon, target_wp['lat'], target_wp['lon'])
            self._send_heading_command(bearing)
            
            # Check if the waypoint has been reached (simple proximity check to change segment)
            dist = GeoUtils.haversine_dist(self.lat, self.lon, target_wp['lat'], target_wp['lon'])
            if dist < 2.0: # If within 2 NM of the waypoint
                 self.current_waypoint_idx += 1
                 # If we change, maybe re-command the heading to the new WP?
                 if self.current_waypoint_idx < len(self.nominal_route):
                      new_wp = self.nominal_route[self.current_waypoint_idx]
                      new_b = GeoUtils.bearing(self.lat, self.lon, new_wp['lat'], new_wp['lon'])
                      self._send_heading_command(new_b)

    def _wait_sim_time(self, duration_sec):
        """
        Waits for a specified duration in X-Plane simulation time.
        Handles mock client for testing.
        """
        if not self.xpc:
             return
        
        current_sim_time = self.xpc.getDREF("sim/time/total_flight_time_sec")[0]
        target_time = current_sim_time + duration_sec
        
        start_real = time.time()
        # Allow waiting up to duration + buffer, assuming sim speed >= 1x
        # If paused, this will still timeout eventually to unblock
        timeout = duration_sec * 2.0 + 10.0 
        
        while current_sim_time < target_time:
             if self.xpc.__class__.__name__ == 'MockXPlaneConnect':
                 # Fast-forward mock time
                 self.xpc.drefs["sim/time/total_flight_time_sec"] += duration_sec
                 pass # In mock, we can just advance time and break
             
             if time.time() - start_real > timeout:
                  # Timeout reached - sim probably paused or too slow
                  break
                  
             time.sleep(1.0) # Check every second
             current_sim_time = self.xpc.getDREF("sim/time/total_flight_time_sec")[0]

    def _sample_wind_at(self, lat, lon, alt):
        """
        Samples wind at an arbitrary position.
        Currently, returns the global wind for simplicity.
        """
        return self.wind_u, self.wind_v

    def _calculate_xte(self):
        """
        Calculates the Cross-Track Error (XTE) relative to the current segment.
        """
        if self.current_waypoint_idx >= len(self.nominal_route):
             return 0.0 # No XTE if route is completed
             
        target_wp = self.nominal_route[self.current_waypoint_idx]
        
        # For XTE, we need the previous point on the nominal route
        if self.current_waypoint_idx == 0:
            return 0.0
        
        prev_wp = self.nominal_route[self.current_waypoint_idx - 1]
        
        return GeoUtils.cross_track_error(
            self.lat, self.lon, 
            prev_wp['lat'], prev_wp['lon'], 
            target_wp['lat'], target_wp['lon']
        )

    def close(self):
        if self.xpc:
            self.xpc.close()
