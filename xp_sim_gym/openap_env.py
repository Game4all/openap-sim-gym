import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from openap import FuelFlow
from xp_sim_gym.config import PlaneConfig, EnvironmentConfig
from .utils import GeoUtils

from .constants import (
    FT_TO_M, KTS_TO_M_S, MAX_DEVIATION_SEGMENTS, MAX_XTRACK_ERROR_NM,
    MAX_ALT, MAX_SPD, MAX_FUEL, MAX_WIND, MAX_DIST,
    MIN_HEADING_OFFSET, MAX_HEADING_OFFSET,
    MIN_DURATION_MIN, MAX_DURATION_MIN, NM_TO_METER, STD_RATE_TURN_DEG_PER_SEC
)


class OpenAPNavEnv(gym.Env):
    """
    Un environement gymnasium utilisant OpenAP pour simuler un avion pour apprendre à naviguer en conditions de vent. 

    **Observations:**
    Vecteur numpy avec `9 + 4 + (4 * lookahead_count)`.
    *Note : Toutes les altitudes sont en **mètres**.*

    1.  **État de l'Avion (9)** :
        *   `norm_alt` : Altitude actuelle / MAX_ALT (45,000 ft)
        *   `norm_tas` : Vitesse Vraie (TAS) / MAX_SPD (600 kts)
        *   `norm_gs` : Vitesse Sol (GS) / MAX_SPD (600 kts)
        *   `norm_fuel` : Quantité de Carburant / MAX_FUEL (20,000 kg)
        *   `norm_wfwd` : Vent longitudinal relatif à l'avion / MAX_WIND
        *   `norm_wrgt` : Vent latéral relatif à l'avion / MAX_WIND
        *   `applied_offset` : La dernière action de déviation demandée (normalisée [-1, 1]).
        *   `applied_duration` : La dernière durée demandée (normalisée [-1, 1]).
        *   `norm_auto_heading` : Le cap cible du pilote automatique (cap vers le prochain waypoint), normalisé [-1, 1].

    2.  **Contexte de la Route (4)** :
        *   `xte` : Erreur Latérale (Cross-Track Error) / MAX_XTRACK_ERROR_NM (50 NM)
        *   `dist_to_wpt` : Distance au prochain waypoint / MAX_DIST (1000 NM)
        *   `track_angle_error` : Erreur d'angle de route (Track vs Bearing) / 180 degrés.
        *   `dist_to_dest` : Distance à la destination / (2 * MAX_DIST)

    3.  **Anticipation (Lookahead) (4 * N)** : 
        Pour chacun des `N` (par défaut 3) prochains waypoints :
        *   `d` : Distance depuis l'avion / MAX_DIST
        *   `rel_b` : Relèvement relatif / 180 degrés
        *   `along` : Composante de vent longitudinale / MAX_WIND
        *   `cross` : Composante de vent latérale / MAX_WIND

    **Actions:**
    Un vecteur de 2 éléments normalisé entre [-1.0, 1.0] :
    1.  `Heading Offset` : Déviation par rapport au cap **autonome** (vers le prochain waypoint). 
        *   0 => Voler directement vers le waypoint.
        *   Normalisé sur [MIN_HEADING_OFFSET, MAX_HEADING_OFFSET] degrés.
    2.  `Duration` : Durée de l'action avant le prochain pas de décision.
        *   Normalisé sur [MIN_DURATION_MIN, MAX_DURATION_MIN] minutes.

    La simulation utilise des sous-pas d'une minute pour la physique.
    """
    metadata = {"render_modes": [], "render_fps": 1}

    def __init__(self, plane_config: PlaneConfig, env_config: EnvironmentConfig, verbose: bool = False):
        super().__init__()

        self.plane_config = plane_config
        self.env_config = env_config
        self.verbose = verbose

        self.aircraft_type = self.plane_config.aircraft_type
        self.fuel_flow_model = FuelFlow(ac=self.aircraft_type)

        self.lookahead_count = self.env_config.lookahead_count
        self.current_waypoint_idx = 0
        self.nominal_route = []
        self.wind_streams = []

        # Obs dim: 9 (State) + 4 (Route) + 4*N (Lookahead)
        obs_dim = 9 + 4 + (4 * self.lookahead_count)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Normalized action space [-1, 1] for 2 components:
        # 1. Heading Offset [MIN_HEADING_OFFSET, MAX_HEADING_OFFSET]
        # 2. Duration [MIN_DURATION_MIN, MAX_DURATION_MIN] minutes
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def _setup_initial_state(self):
        """Setup the initial state of the OpenAP simulation."""
        self.steps_taken = 0
        self.current_waypoint_idx = 1
        self.previous_offset = 0.0
        self.previous_duration = 0.0
        self.segment_durations = [0.0, 0.0, 0.0]
        self.all_segment_durations = []

        if not self.nominal_route:
            assert self.nominal_route, "Nominal route must be set (via set_nominal_route) before reset/step"

        start_node = self.nominal_route[0]
        self.lat = start_node.get('lat', self.plane_config.initial_lat)
        self.lon = start_node.get('lon', self.plane_config.initial_lon)
        self.alt_m = start_node.get('alt', self.plane_config.initial_alt_m)

        if self.plane_config.initial_heading_mag is not None:
            self.heading_mag = self.plane_config.initial_heading_mag
        elif len(self.nominal_route) > 1:
            next_node = self.nominal_route[1]
            self.heading_mag = GeoUtils.bearing(
                self.lat, self.lon, next_node['lat'], next_node['lon'])
        else:
            self.heading_mag = 0.0

        self.current_fuel_kg = self.plane_config.initial_fuel_kg
        self.tas_ms = self.plane_config.initial_tas_ms
        self.gs_ms = self.tas_ms

        # Set initial local wind
        self.wind_u, self.wind_v = self._sample_wind_at(
            self.lat, self.lon, self.alt_m)

    def set_nominal_route(self, route: list):
        """Sets the nominal route to follow."""
        self.nominal_route = route

    def set_wind_config(self, wind_streams: list):
        """Sets the wind configuration."""
        self.wind_streams = wind_streams

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_initial_state()
        return self._get_observation(), {"segment_durations": [0.0, 0.0, 0.0], "all_segment_durations": []}

    def step(self, action):
        # Capture state before simulation for progression reward
        prev_total_dist = self._get_total_remaining_dist()

        # action is in [-1, 1], map to physical ranges
        # heading offset mapping: [-1, 1] -> [MIN_HEADING_OFFSET, MAX_HEADING_OFFSET]
        heading_offset_deg = MIN_HEADING_OFFSET + \
            (action[0] + 1.0) * 0.5 * (MAX_HEADING_OFFSET - MIN_HEADING_OFFSET)

        # duration mapping: [-1, 1] -> [MIN_DURATION_MIN, MAX_DURATION_MIN]
        duration_min = MIN_DURATION_MIN + \
            (action[1] + 1.0) * 0.5 * (MAX_DURATION_MIN - MIN_DURATION_MIN)
        duration_sec = duration_min * 60.0

        # is_deviating is now true if heading offset is non-zero (with small epsilon)
        is_deviating = abs(heading_offset_deg) > 0.01

        # Store action for observation
        self.previous_offset = action[0]
        self.previous_duration = action[1]
        self.steps_taken += 1

        # --- Physics Integration ---
        fuel_consumed, total_distance_m = self._integrate_physics(
            duration_sec, heading_offset_deg)

        # 5. Check Waypoint/Segment Logic
        was_complete = self.current_waypoint_idx >= len(self.nominal_route)
        self._check_waypoint_progression()
        is_complete = self.current_waypoint_idx >= len(self.nominal_route)

        # 6. Reward Calculation and Status
        xte_nm = self._calculate_xte()
        ate_nm = self._calculate_atd()

        # Calculate Global Progression (Reward for reducing total distance to destination)
        new_total_dist = self._get_total_remaining_dist()
        progression_nm = prev_total_dist - new_total_dist

        # Check for first-time route completion
        terminal_bonus = is_complete and not was_complete

        # Calculate VMG Gain (kts)
        vmg_kts = (progression_nm / duration_min) * 60.0
        tas_kts = self.tas_ms / KTS_TO_M_S
        vmg_gain = vmg_kts - tas_kts

        reward = self._compute_reward(
            fuel_consumed=fuel_consumed,
            duration_min=duration_min,
            xte_nm=xte_nm,
            progression_nm=progression_nm,
            terminal_bonus=terminal_bonus,
            vmg_gain=vmg_gain,
            is_deviating=is_deviating
        )

        terminated = False
        truncated = False

        if self.steps_taken >= self.env_config.max_steps:
            truncated = True

        if abs(xte_nm) > MAX_XTRACK_ERROR_NM:
            terminated = True

        if is_complete:
            terminated = True

        info = {
            "xte": xte_nm,
            "ate": ate_nm,
            "fuel_consumed": fuel_consumed,
            "progression": progression_nm,
            "distance_flown": total_distance_m / NM_TO_METER,
            "duration": duration_min,
            "gs_ms": self.gs_ms,
            "segment_durations": self.segment_durations.copy(),
            "all_segment_durations": self.all_segment_durations.copy()
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _integrate_physics(self, duration_sec, heading_offset_deg):
        """
        Integrates the flight physics for a given duration.
        Returns:
            fuel_consumed (kg)
            total_distance_m (meters flown)
        """
        remaining_time = duration_sec

        dt_sim = 60.0
        turn_rate_deg_min = STD_RATE_TURN_DEG_PER_SEC * dt_sim

        total_fuel_consumed = 0.0
        total_distance_m = 0.0

        while remaining_time > 0:
            current_dt = min(dt_sim, remaining_time)

            # Update wind at current position
            self.wind_u, self.wind_v = self._sample_wind_at(
                self.lat, self.lon, self.alt_m)

            # a. Calculate Auto Heading (Dynamic per sub-step)
            if self.current_waypoint_idx < len(self.nominal_route):
                target_wp = self.nominal_route[self.current_waypoint_idx]
                auto_heading = GeoUtils.bearing(
                    self.lat, self.lon, target_wp['lat'], target_wp['lon'])
            else:
                auto_heading = self.heading_mag

            # b. Determine Target Heading
            target_heading = (auto_heading + heading_offset_deg) % 360.0

            # c. Apply Rate Limit to Heading Change
            # Calculate smallest difference
            diff = (target_heading - self.heading_mag + 180) % 360 - 180

            # Max change for this timestep
            max_change = turn_rate_deg_min * current_dt

            if abs(diff) <= max_change:
                self.heading_mag = target_heading
            else:
                self.heading_mag += math.copysign(max_change, diff)
                self.heading_mag %= 360.0

            # d. Kinematics (for current_dt)
            heading_rad = math.radians(self.heading_mag)

            # TAS vector
            tas_n = self.tas_ms * math.cos(heading_rad)
            tas_e = self.tas_ms * math.sin(heading_rad)

            # Ground Speed vector
            gs_n = tas_n + self.wind_v
            gs_e = tas_e + self.wind_u

            self.gs_ms = math.sqrt(gs_e**2 + gs_n**2)
            track_rad = math.atan2(gs_e, gs_n)

            # Distance flown in this sub-step
            dist_m = self.gs_ms * current_dt
            dist_nm = dist_m / NM_TO_METER
            total_distance_m += dist_m

            # Update Position
            delta_lat = (dist_nm * math.cos(track_rad)) / 60.0
            self.lat += delta_lat

            delta_lon = (dist_nm * math.sin(track_rad)) / \
                (60.0 * math.cos(math.radians(self.lat)))
            self.lon += delta_lon

            # e. Fuel Consumption (for current_dt)
            ff_kg_s = self.fuel_flow_model.enroute(
                mass=self.current_fuel_kg + 40000,
                tas=self.tas_ms / KTS_TO_M_S,
                alt=self.alt_m / FT_TO_M,
            )
            fuel_step = ff_kg_s * current_dt
            self.current_fuel_kg -= fuel_step
            total_fuel_consumed += fuel_step
            self.segment_durations[2] += (current_dt / 60.0)

            # f. Check Waypoint Passing
            self._check_waypoint_progression()

            remaining_time -= current_dt

        return total_fuel_consumed, total_distance_m

    def _compute_reward(self, fuel_consumed, duration_min, xte_nm, progression_nm, terminal_bonus, vmg_gain, is_deviating):
        """
        Optimized Reward Function for Efficiency (Min Fuel/Time).
        """

        # --- Weights ---
        # 1. Progression: The main driver. High enough to overcome small XTE penalties.
        W_PROGRESS = 1.0

        # 2. Fuel/Time: The cost of existence.
        # This implicitly penalizes adding distance. If you fly longer, you lose more points.
        W_FUEL = -0.1          # e.g., -100kg = -10 pts

        # 3. XTE: Safety Constraint.
        # Use a "Corridor" approach: Low penalty for small deviations (allows optimization),
        # high penalty for large deviations.
        W_XTE_BASE = -0.05
        W_XTE_HIGH = -0.5

        # 4. Action Smoothing
        W_DEVIATION = -0.2     # Cost of deviating from the nominal course

        # --- 1. Progression (The Carrot) ---
        # Reward simply getting closer.
        reward_progress = W_PROGRESS * progression_nm

        # --- 2. Cost of Operation (The Stick for adding distance) ---
        # If the agent takes a detour that adds 10km, it burns more fuel.
        # This term AUTOMATICALLY penalizes inefficient routes.
        reward_fuel = W_FUEL * fuel_consumed

        # --- 3. Efficiency Bonus (VMG) ---
        # Optional: Boost learning by explicitly rewarding flying EFFICIENTLY.
        # If VMG > TAS, the agent is using wind effectively towards the target.
        # This replaces your old 'Tailwind' reward but accounts for direction.
        reward_efficiency = 0.0
        if vmg_gain > 0:
            reward_efficiency = 0.75 * vmg_gain  # Bonus for super-efficiency
        else:
            reward_efficiency = 1.0 * vmg_gain  # Penalty for fighting wind or bad heading

        # Scale by duration to make it physically consistent
        reward_efficiency *= (duration_min / 10.0)

        # --- 5. Deviation Penalty (The "No Reason" Stick) ---
        # Discourage deviation unless it's worth it.
        reward_deviation = 0.0
        if is_deviating:
            # We penalize based on the magnitude of the deviation requested.
            # self.previous_offset is the normalized [-1, 1] action.
            reward_deviation = W_DEVIATION * \
                abs(self.previous_offset) * duration_min

        # --- 6. Smart XTE Penalty ---
        # Autoriser la déviation jusqu'à 10NM puis pénaliser plus amplement à partir de ce seuil
        xte_thresh = self.env_config.flyby_waypoint_dist * 0.5
        if abs(xte_nm) < xte_thresh:
            reward_xte = W_XTE_BASE * abs(xte_nm) * duration_min
        else:
            # Exponentially harder penalty beyond 10 NM
            reward_xte = (W_XTE_BASE * xte_thresh + W_XTE_HIGH *
                          (abs(xte_nm) - xte_thresh)) * duration_min

        # --- 7. Terminal ---
        terminal_reward = 0.0
        if abs(xte_nm) > MAX_XTRACK_ERROR_NM:
            terminal_reward = -100.0  # Crash penalty

        # Total
        step_reward = (
            reward_progress +
            reward_fuel +
            reward_xte +
            reward_efficiency +
            reward_deviation
        )

        return step_reward + terminal_reward

    def _check_waypoint_progression(self):
        """
        Sequences waypoints if the aircraft gets close (15 NM) OR passes abeam 
        (along-track distance to the next waypoint becomes negative or very small).
        """
        if self.current_waypoint_idx < len(self.nominal_route):
            target_wp = self.nominal_route[self.current_waypoint_idx]

            # 1. Proximity check
            dist = GeoUtils.haversine_dist(
                self.lat, self.lon, target_wp['lat'], target_wp['lon'])

            # 2. Abeam check (if we have a previous waypoint to define a segment)
            passed_abeam = False
            if self.current_waypoint_idx > 0:
                prev_wp = self.nominal_route[self.current_waypoint_idx - 1]
                atd = GeoUtils.along_track_distance(
                    self.lat, self.lon,
                    prev_wp['lat'], prev_wp['lon'],
                    target_wp['lat'], target_wp['lon']
                )
                segment_dist = GeoUtils.haversine_dist(
                    prev_wp['lat'], prev_wp['lon'],
                    target_wp['lat'], target_wp['lon']
                )
                # If along-track distance exceeds segment length, we've passed it
                if atd > segment_dist:
                    passed_abeam = True

            if dist < self.env_config.flyby_waypoint_dist or passed_abeam:
                # Record duration
                self.all_segment_durations.append(self.segment_durations[2])
                
                # Shift durations: [d1, d2, d3] -> [d2, d3, 0.0]
                self.segment_durations[0] = self.segment_durations[1]
                self.segment_durations[1] = self.segment_durations[2]
                self.segment_durations[2] = 0.0
                self.current_waypoint_idx += 1
                # If we passed one, recursively check if we passed the next one too
                self._check_waypoint_progression()

    def _calculate_xte(self):
        """Calcule la XTE aka. Cross-Track Error (déviation par rapport au plan de vol)"""
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

    def _calculate_atd(self):
        """Calcule la distance Along-Track (ATD) par rapport au segment actuel."""
        if self.current_waypoint_idx >= len(self.nominal_route) or self.current_waypoint_idx == 0:
            return 0.0

        target_wp = self.nominal_route[self.current_waypoint_idx]
        prev_wp = self.nominal_route[self.current_waypoint_idx - 1]

        return GeoUtils.along_track_distance(
            self.lat, self.lon,
            prev_wp['lat'], prev_wp['lon'],
            target_wp['lat'], target_wp['lon']
        )

    def _get_total_remaining_dist(self):
        """
        Calcule la distance totale restante le long de la route programmée.
        Distance = Distance (avion -> waypoint actuel) + Somme des legs restants.
        """
        if self.current_waypoint_idx >= len(self.nominal_route):
            return 0.0

        # 1. Distance to current active waypoint
        target_wp = self.nominal_route[self.current_waypoint_idx]
        total_dist = GeoUtils.haversine_dist(
            self.lat, self.lon, target_wp['lat'], target_wp['lon'])

        # 2. Add all subsequent legs
        for i in range(self.current_waypoint_idx, len(self.nominal_route) - 1):
            wp1 = self.nominal_route[i]
            wp2 = self.nominal_route[i+1]
            total_dist += GeoUtils.haversine_dist(
                wp1['lat'], wp1['lon'], wp2['lat'], wp2['lon'])

        return total_dist

    def _sample_wind_at(self, lat, lon, alt):
        """Samples the total wind vector from all active wind streams at the given location."""
        total_u, total_v = 0.0, 0.0

        for stream in self.wind_streams:
            # 1. Project stream 'end' point effectively far away to define the line
            # 1 deg ~ 60 NM. We project 1000 NM out.
            d_lat = (1000.0 / 60.0) * math.cos(math.radians(stream.direction))
            d_lon = (1000.0 / 60.0) * math.sin(math.radians(stream.direction)
                                               ) / math.cos(math.radians(stream.lat))

            end_lat, end_lon = stream.lat + d_lat, stream.lon + d_lon

            # 2. Calculate cross-track distance (perpendicular distance to stream core)
            xtd_nm = GeoUtils.cross_track_error(
                lat, lon,
                stream.lat, stream.lon,
                end_lat, end_lon
            )

            # 3. Gaussian intensity profile
            # Speed = MaxSpeed * exp( - distance^2 / (2 * width^2) )
            # width is sigma (standard deviation)
            intensity = math.exp(- (xtd_nm**2) / (2 * stream.width_nm**2))

            if intensity > 0.01:  # Optimization cutoff
                wind_speed = stream.max_speed_kts * intensity

                # Convert to components (Weather convention: direction is WHERE wind comes FROM?
                # Wait, usually for data we use U/V flow components (where it goes TO).
                # WindStreamConfig.direction is likely "Direction of flow" (River model).
                # So if direction is 90 (East), U is +, V is 0.

                rad = math.radians(stream.direction)
                # U = Speed * sin(dir), V = Speed * cos(dir)
                u = wind_speed * math.sin(rad) * KTS_TO_M_S  # to m/s
                v = wind_speed * math.cos(rad) * KTS_TO_M_S

                total_u += u
                total_v += v

        return total_u, total_v

    def _get_observation(self):
        state_obs = self._get_state_obs()
        route_obs = self._get_route_obs()
        lookahead_obs = self._get_lookahead_obs()

        return np.concatenate([state_obs, route_obs, lookahead_obs])

    def _get_state_obs(self):
        """Constructs the aircraft state observation."""
        # altitudes are in meters
        norm_alt = self.alt_m / (MAX_ALT * FT_TO_M)
        norm_tas = self.tas_ms / (MAX_SPD * KTS_TO_M_S)
        norm_gs = self.gs_ms / (MAX_SPD * KTS_TO_M_S)
        norm_fuel = self.current_fuel_kg / MAX_FUEL

        # Plane-local wind vector (Forward, Right)
        heading_rad = math.radians(self.heading_mag)
        w_fwd = self.wind_u * \
            math.sin(heading_rad) + self.wind_v * math.cos(heading_rad)
        w_rgt = self.wind_u * \
            math.cos(heading_rad) - self.wind_v * math.sin(heading_rad)
        norm_wfwd = w_fwd / (MAX_WIND * KTS_TO_M_S)
        norm_wrgt = w_rgt / (MAX_WIND * KTS_TO_M_S)

        # New State Observation: Applied Offset
        # We use the raw action value from the previous step which is already in [-1, 1]
        norm_offset = self.previous_offset

        # Calculate AP target heading (auto_heading)
        if self.current_waypoint_idx < len(self.nominal_route):
            target_wp = self.nominal_route[self.current_waypoint_idx]
            auto_heading = GeoUtils.bearing(
                self.lat, self.lon, target_wp['lat'], target_wp['lon'])
        else:
            auto_heading = self.heading_mag

        # Normalize Auto Heading to [-1, 1]
        norm_auto_heading = (auto_heading - 180.0) / 180.0

        # Clip state observations to [-1, 1] or [0, 1]
        return np.array([
            np.clip(norm_alt, 0.0, 1.0),
            np.clip(norm_tas, 0.0, 1.0),
            np.clip(norm_gs, 0.0, 1.0),
            np.clip(norm_fuel, 0.0, 1.0),
            np.clip(norm_wfwd, -1.0, 1.0),
            np.clip(norm_wrgt, -1.0, 1.0),
            np.clip(norm_offset, -1.0, 1.0),
            np.clip(self.previous_duration, -1.0, 1.0),
            np.clip(norm_auto_heading, -1.0, 1.0)
        ], dtype=np.float32)

    def _get_route_obs(self):
        """Constructs the route context observation."""
        if self.current_waypoint_idx < len(self.nominal_route):
            target_wp = self.nominal_route[self.current_waypoint_idx]
            if self.current_waypoint_idx == 0:
                prev_lat, prev_lon = self.lat, self.lon
            else:
                prev_wp = self.nominal_route[self.current_waypoint_idx - 1]
                prev_lat, prev_lon = prev_wp['lat'], prev_wp['lon']

            target_lat, target_lon = target_wp['lat'], target_wp['lon']

            xte = GeoUtils.cross_track_error(
                self.lat, self.lon, prev_lat, prev_lon, target_lat, target_lon)
            dist_to_wpt = GeoUtils.haversine_dist(
                self.lat, self.lon, target_lat, target_lon)

            # Replaced bearing error with Track Angle Error
            # Nominal leg bearing
            leg_bearing = GeoUtils.bearing(
                prev_lat, prev_lon, target_lat, target_lon)

            # Actual Track
            gs_n = self.tas_ms * \
                math.cos(math.radians(self.heading_mag)) + self.wind_v
            gs_e = self.tas_ms * \
                math.sin(math.radians(self.heading_mag)) + self.wind_u
            current_track = (math.degrees(math.atan2(gs_e, gs_n)) + 360) % 360

            track_err = (current_track - leg_bearing + 180) % 360 - 180

            last_wp = self.nominal_route[-1]
            dist_to_dest = GeoUtils.haversine_dist(
                self.lat, self.lon, last_wp['lat'], last_wp['lon'])

        else:
            xte = 0.0
            dist_to_wpt = 0.0
            track_err = 0.0
            dist_to_dest = 0.0

        return np.array([
            xte / MAX_XTRACK_ERROR_NM,
            dist_to_wpt / MAX_DIST,
            track_err / 180.0,
            dist_to_dest / (MAX_DIST * 2)
        ], dtype=np.float32)

    def _get_lookahead_obs(self):
        """Constructs the lookahead observation for future waypoints."""
        lookahead_obs = []
        for i in range(self.lookahead_count):
            idx = self.current_waypoint_idx + i
            if idx < len(self.nominal_route):
                wp = self.nominal_route[idx]
                d = GeoUtils.haversine_dist(
                    self.lat, self.lon, wp['lat'], wp['lon'])
                b = GeoUtils.bearing(self.lat, self.lon, wp['lat'], wp['lon'])
                rel_b = (b - self.heading_mag + 180) % 360 - 180

                w_u, w_v = self._sample_wind_at(
                    wp['lat'], wp['lon'], wp.get('alt', 33000))

                if idx == 0:
                    leg_brg = GeoUtils.bearing(
                        self.lat, self.lon, wp['lat'], wp['lon'])
                else:
                    p_wp = self.nominal_route[idx-1]
                    leg_brg = GeoUtils.bearing(
                        p_wp['lat'], p_wp['lon'], wp['lat'], wp['lon'])

                leg_rad = math.radians(leg_brg)
                along = w_u * math.sin(leg_rad) + w_v * math.cos(leg_rad)
                cross = w_u * math.cos(leg_rad) - w_v * math.sin(leg_rad)

                lookahead_obs.extend([
                    d / MAX_DIST,
                    rel_b / 180.0,
                    along / (MAX_WIND * KTS_TO_M_S),
                    cross / (MAX_WIND * KTS_TO_M_S)
                ])
            else:
                lookahead_obs.extend([1.0, 0.0, 0.0, 0.0])

        return np.array(lookahead_obs, dtype=np.float32)
