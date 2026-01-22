import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from openap import FuelFlow
from xp_sim_gym.config import PlaneEnvironmentConfig, WindStreamConfig
from .utils import GeoUtils
from .route_generator import RouteStageGenerator
from .constants import (
    MAX_DEVIATION_SEGMENTS, MAX_XTRACK_ERROR_NM,
    MAX_ALT, MAX_SPD, MAX_FUEL, MAX_WIND, MAX_DIST,
    MIN_HEADING_OFFSET, MAX_HEADING_OFFSET,
    MIN_DURATION_MIN, MAX_DURATION_MIN
)


class OpenAPNavEnv(gym.Env):
    """
    Un environement gymnasium utilisant OpenAP pour simuler un avion. 
    Cet environement vise à permettre le pré-entraînement d'un agent pour optimiser les déviations de vol.

    **Observations:**
    Vecteur numpy avec `7 + 4 + (4 * lookahead_count)`.
    *Note : Toutes les altitudes sont en **mètres**.*

    1.  **État de l'Avion (7)** :
        *   `norm_alt` : Altitude actuelle / MAX_ALT (45,000 ft)
        *   `norm_tas` : Vitesse Vraie (TAS) / MAX_SPD (600 kts)
        *   `norm_gs` : Vitesse Sol (GS) / MAX_SPD (600 kts)
        *   `norm_fuel` : Quantité de Carburant / MAX_FUEL (20,000 kg)
        *   `norm_wfwd` : Vent longitudinal relatif à l'avion / MAX_WIND
        *   `norm_wrgt` : Vent latéral relatif à l'avion / MAX_WIND
        *   `applied_offset` : La dernière action de déviation demandée (normalisée [-1, 1]).

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
    2.  `Duration` : Durée de l'action de maintien de cap.
        *   Normalisé sur [MIN_DURATION_MIN, MAX_DURATION_MIN] minutes.
    3.  `Deviate` : Toggle d'activation de la déviation.
        *   Action > 0 => Déviation activée (utilise Heading Offset).
        *   Action <= 0 => Déviation désactivée (Heading Offset forcé à 0).
    """
    metadata = {"render_modes": [], "render_fps": 1}

    def __init__(self, config: PlaneEnvironmentConfig):
        super().__init__()

        assert config is not None, "Config should not be none"

        self.config = config

        self.aircraft_type = self.config.aircraft_type
        self.fuel_flow_model = FuelFlow(ac=self.aircraft_type)

        self.lookahead_count = self.config.lookahead_count
        self.stage = 1
        self.current_waypoint_idx = 0
        self.nominal_route = self.config.nominal_route if self.config.nominal_route else []

        # Obs dim: 7 (State) + 4 (Route) + 4*N (Lookahead)
        obs_dim = 7 + 4 + (4 * self.lookahead_count)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Normalized action space [-1, 1] for 3 components
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.previous_offset = 0.0  # Initialize previous offset
        self.previous_is_deviating = False  # Track deviation state changes

        self.route_generator = RouteStageGenerator(self.config)

        self._setup_initial_state()

    def _setup_initial_state(self):
        """Setup l'état initial de la simulation OpenAP"""
        self.steps_taken = 0
        self.current_waypoint_idx = 1

        # Use config route if provided, otherwise generate a fresh one for the current stage
        if self.config.nominal_route:
            self.nominal_route = self.config.nominal_route
        else:
            self.nominal_route = self.route_generator.generate_route(
                self.stage)

        if self.nominal_route:
            start_node = self.nominal_route[0]
            self.lat = start_node.get('lat', self.config.initial_lat)
            self.lon = start_node.get('lon', self.config.initial_lon)
            self.alt_m = start_node.get('alt', self.config.initial_alt_m)

            if self.config.initial_heading_mag is not None:
                self.heading_mag = self.config.initial_heading_mag
            elif len(self.nominal_route) > 1:
                next_node = self.nominal_route[1]
                self.heading_mag = GeoUtils.bearing(
                    self.lat, self.lon, next_node['lat'], next_node['lon'])
            else:
                self.heading_mag = 0.0
        else:
            self.lat = self.config.initial_lat
            self.lon = self.config.initial_lon
            self.alt_m = self.config.initial_alt_m
            self.heading_mag = self.config.initial_heading_mag if self.config.initial_heading_mag is not None else 0.0

        self.current_fuel_kg = self.config.initial_fuel_kg
        self.tas_ms = self.config.initial_tas_ms
        self.gs_ms = self.tas_ms

        # Important: generate wind AFTER route so we can align it
        self.wind_streams = self.route_generator.generate_wind(
            stage=self.stage,
            route=self.nominal_route,
            start_lat=self.lat,
            start_lon=self.lon
        )

        # Set initial local wind
        self.wind_u, self.wind_v = self._sample_wind_at(
            self.lat, self.lon, self.alt_m)

    def set_pretraining_stage(self, stage):
        """Règle le niveau de difficulté de l'environnement pour le pré-entraînement"""
        if stage not in [1, 2, 3, 4, 5]:
            print(f"Warning: Etape {stage} invalide.")
            return

        self.stage = stage
        print(f"OpenAPNavEnv switched to Stage {stage}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_initial_state()
        self.previous_is_deviating = False
        return self._get_observation(), {}

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

        # Deviate Toggle: action[2] > 0 means we allow deviation
        is_deviating = action[2] > 0
        if not is_deviating:
            heading_offset_deg = 0.0

        remaining_time = duration_sec
        dt_sim = 60.0  # 1 minute sub-steps for waypoint checking

        total_fuel_consumed = 0.0
        total_distance_m = 0.0

        # Store action for observation
        self.previous_offset = action[0]

        total_gs_weighted = 0.0
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

            # b. Apply Action Offset to current Auto Heading
            target_heading = (auto_heading + heading_offset_deg) % 360.0

            # c. Kinematics (for current_dt)
            heading_rad = math.radians(target_heading)

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
            dist_nm = dist_m / 1852.0
            total_distance_m += dist_m

            # Update Position
            delta_lat = (dist_nm * math.cos(track_rad)) / 60.0
            self.lat += delta_lat

            delta_lon = (dist_nm * math.sin(track_rad)) / \
                (60.0 * math.cos(math.radians(self.lat)))
            self.lon += delta_lon

            # Update Heading
            self.heading_mag = target_heading

            total_gs_weighted += self.gs_ms * current_dt

            # d. Fuel Consumption (for current_dt)
            ff_kg_s = self.fuel_flow_model.enroute(
                mass=self.current_fuel_kg + 40000,
                tas=self.tas_ms / 0.514444,
                alt=self.alt_m / 0.3048,
            )
            fuel_step = ff_kg_s * current_dt
            self.current_fuel_kg -= fuel_step
            total_fuel_consumed += fuel_step

            # e. Check Waypoint Passing
            self._check_waypoint_progression()

            remaining_time -= current_dt

        fuel_consumed = total_fuel_consumed

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
        # VMG is the rate of progress towards the destination
        vmg_kts = (progression_nm / duration_min) * 60.0
        tas_kts = self.tas_ms / 0.514444
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

        if self.steps_taken >= MAX_DEVIATION_SEGMENTS:
            truncated = True

        if abs(xte_nm) > MAX_XTRACK_ERROR_NM:
            terminated = True

        if is_complete:
            terminated = True

        self.steps_taken += 1
        self.previous_is_deviating = is_deviating

        info = {
            "xte": xte_nm,
            "ate": ate_nm,
            "fuel_consumed": fuel_consumed,
            "progression": progression_nm,
            "distance_flown": total_distance_m / 1852.0
        }

        return self._get_observation(), reward, terminated, truncated, info

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
        W_DEV_CHANGE = -0.1    # Penalty for flickering the deviation switch

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
            reward_efficiency = 0.5 * vmg_gain  # Bonus for super-efficiency
        else:
            reward_efficiency = 1.0 * vmg_gain  # Penalty for fighting wind or bad heading

        # Scale by duration to make it physically consistent
        reward_efficiency *= (duration_min / 10.0)

        # --- 4. Smart XTE Penalty ---
        # Allow deviation up to 10 NM with low penalty, then scale up.
        # This allows the agent to leave the line to find wind without immediate panic.
        if abs(xte_nm) < 10.0:
            reward_xte = W_XTE_BASE * abs(xte_nm) * duration_min
        else:
            # Exponentially harder penalty beyond 10 NM
            reward_xte = (W_XTE_BASE * 10.0 + W_XTE_HIGH * (abs(xte_nm) - 10.0)) * duration_min

        # --- 5. Action Smoothing (Optional) ---
        reward_smoothing = 0.0
        if is_deviating != self.previous_is_deviating:
            reward_smoothing = W_DEV_CHANGE

        # --- 6. Terminal ---
        terminal_reward = 0.0
        if terminal_bonus:
            terminal_reward = 100.0  # Big finish reward
        elif abs(xte_nm) > MAX_XTRACK_ERROR_NM:
            terminal_reward = -100.0 # Crash penalty

        # Total
        step_reward = (
            reward_progress +
            reward_fuel +
            reward_xte +
            reward_efficiency +
            reward_smoothing
        )

        return step_reward + terminal_reward

    def _check_waypoint_progression(self):
        """
        Sequences waypoints if the aircraft gets close (5 NM) OR passes abeam 
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

            if dist < 5.0 or passed_abeam:
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
                u = wind_speed * math.sin(rad) * 0.514444  # to m/s
                v = wind_speed * math.cos(rad) * 0.514444

                total_u += u
                total_v += v

        return total_u, total_v

    def _get_observation(self):
        # altitudes are in meters
        norm_alt = self.alt_m / (MAX_ALT * 0.3048)
        norm_tas = self.tas_ms / (MAX_SPD * 0.514444)
        norm_gs = self.gs_ms / (MAX_SPD * 0.514444)
        norm_fuel = self.current_fuel_kg / MAX_FUEL

        # Plane-local wind vector (Forward, Right)
        heading_rad = math.radians(self.heading_mag)
        w_fwd = self.wind_u * \
            math.sin(heading_rad) + self.wind_v * math.cos(heading_rad)
        w_rgt = self.wind_u * \
            math.cos(heading_rad) - self.wind_v * math.sin(heading_rad)
        norm_wfwd = w_fwd / (MAX_WIND * 0.514444)
        norm_wrgt = w_rgt / (MAX_WIND * 0.514444)

        # New State Observation: Applied Offset
        # We use the raw action value from the previous step which is already in [-1, 1]
        norm_offset = self.previous_offset

        # Clip state observations to [-1, 1] or [0, 1]
        state_obs = np.array([
            np.clip(norm_alt, 0.0, 1.0),
            np.clip(norm_tas, 0.0, 1.0),
            np.clip(norm_gs, 0.0, 1.0),
            np.clip(norm_fuel, 0.0, 1.0),
            np.clip(norm_wfwd, -1.0, 1.0),
            np.clip(norm_wrgt, -1.0, 1.0),
            np.clip(norm_offset, -1.0, 1.0)
        ], dtype=np.float32)

        # Route Obs
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
            # We need the track from the previous step which was calculated in step()
            # But here in get_observation we might need to recalculate or store it.
            # Ideally we recalculate current track based on velocity vector
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

        route_obs = np.array([
            xte / MAX_XTRACK_ERROR_NM,
            dist_to_wpt / MAX_DIST,
            track_err / 180.0,
            dist_to_dest / (MAX_DIST * 2)
        ], dtype=np.float32)

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
                    along / (MAX_WIND * 0.5144),
                    cross / (MAX_WIND * 0.5144)
                ])
            else:
                lookahead_obs.extend([1.0, 0.0, 0.0, 0.0])

        return np.concatenate([state_obs, route_obs, np.array(lookahead_obs, dtype=np.float32)])
