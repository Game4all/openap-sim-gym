import gymnasium as gym
import numpy as np
import pygame
import math
import time

from xp_sim_gym.constants import KTS_TO_M_S



class BaseViz:
    """
    Base class for 2D visualization using Pygame.
    Contains shared logic for rendering the map, grid, wind field, and route.
    """

    def __init__(self, width=800, height=800, padding=100):
        self.width = width
        self.height = height
        self.padding = padding

        self.screen = None
        self.clock = None
        self.font = None

        # Scaling Factors (Map Limits)
        self.lon_min = None
        self.lon_max = None
        self.lat_min = None
        self.lat_max = None
        self.show_segment_times = True

    def _calculate_bounds(self, envs_data):
        """
        Calculates lat/lon bounds based on multiple sets of environment data.
        Each data set should contain 'lat', 'lon', 'nominal_route', and 'trajectory'.
        """
        lats = []
        lons = []

        for data in envs_data:
            lats.append(data['lat'])
            lons.append(data['lon'])
            if data.get('nominal_route'):
                lats.extend([wp['lat'] for wp in data['nominal_route']])
                lons.extend([wp['lon'] for wp in data['nominal_route']])
            if data.get('trajectory'):
                lats.extend([p[0] for p in data['trajectory']])
                lons.extend([p[1] for p in data['trajectory']])

        if not lats:
            self.lat_min = self.lat_max = 0
            self.lon_min = self.lon_max = 0
            return

        self.lat_min, self.lat_max = min(lats), max(lats)
        self.lon_min, self.lon_max = min(lons), max(lons)

        # Set a minimum gap to avoid division by zero
        d_lat = max(self.lat_max - self.lat_min, 0.05)
        d_lon = max(self.lon_max - self.lon_min, 0.05)

        # Aspect ratio adjustment
        aspect_ratio = self.width / self.height
        lat_center = (self.lat_min + self.lat_max) / 2
        lon_center = (self.lon_min + self.lon_max) / 2

        cos_lat = math.cos(math.radians(lat_center))

        if (d_lon * cos_lat) / d_lat > aspect_ratio:
            d_lat = (d_lon * cos_lat) / aspect_ratio
        else:
            d_lon = (d_lat * aspect_ratio) / cos_lat

        self.lat_min = lat_center - d_lat * 0.6
        self.lat_max = lat_center + d_lat * 0.6
        self.lon_min = lon_center - d_lon * 0.6
        self.lon_max = lon_center + d_lon * 0.6

    def _to_pixel(self, lat, lon):
        """Converts lat/lon coordinates to screen pixels."""
        if self.lon_max is None or self.lon_min is None or self.lat_max is None or self.lat_min is None:
             return self.width // 2, self.height // 2

        if self.lon_max == self.lon_min or self.lat_max == self.lat_min:
            return self.width // 2, self.height // 2

        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * self.width
        y = self.height - (lat - self.lat_min) / (self.lat_max - self.lat_min) * self.height
        return int(x), int(y)

    def _init_pygame(self, title="Simulation OpenAP"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(title)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)
            self.small_font = pygame.font.SysFont("Arial", 14)

    def _draw_background(self):
        self.screen.fill((20, 22, 28))
        grid_color = (40, 45, 55)
        for lat in np.linspace(self.lat_min, self.lat_max, 10):
            p1 = self._to_pixel(lat, self.lon_min)
            p2 = self._to_pixel(lat, self.lon_max)
            pygame.draw.line(self.screen, grid_color, p1, p2, 1)
        for lon in np.linspace(self.lon_min, self.lon_max, 10):
            p1 = self._to_pixel(self.lat_min, lon)
            p2 = self._to_pixel(self.lat_max, lon)
            pygame.draw.line(self.screen, grid_color, p1, p2, 1)

    def _draw_nominal_route(self, route, current_wp_idx, durations=None, color=(70, 70, 100), label_color=(200, 200, 200), label_offset=(5, 5), draw_route=True, is_delta=False):
        if route:
            points = [self._to_pixel(wp['lat'], wp['lon']) for wp in route]
            if draw_route and len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
            
            for i, p in enumerate(points):
                if draw_route:
                    wp_color = (0, 255, 127) if i == current_wp_idx else (150, 150, 150)
                    pygame.draw.rect(self.screen, wp_color, (p[0]-3, p[1]-3, 6, 6))
                
                # Draw segment duration if available (skip start point)
                if self.show_segment_times and durations and i > 0 and i <= len(durations):
                    dur_val = durations[i-1]
                    
                    final_color = label_color
                    text_content = f"{dur_val:.1f}m"
                    
                    if is_delta:
                        if dur_val > 0.01:
                            final_color = (255, 100, 100) # Red-ish for gain
                            text_content = f"+{dur_val:.1f}m"
                        elif dur_val < -0.01:
                            final_color = (100, 255, 100) # Green-ish for loss
                            text_content = f"{dur_val:.1f}m"
                        else:
                            final_color = (200, 200, 200)

                    dur_text = self.small_font.render(text_content, True, final_color)
                    # Create a semi-transparent surface for the text
                    dur_text.set_alpha(180) 
                    self.screen.blit(dur_text, (p[0] + label_offset[0], p[1] + label_offset[1]))

    def _draw_wind_field(self, env):
        if hasattr(env, 'wind_streams') or hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'wind_streams'):
            # Accessing from unwrapped if it's a wrapper
            inner_env = env.unwrapped if hasattr(env, 'unwrapped') else env
            
            grid_step_lat = (self.lat_max - self.lat_min) / 10.0
            grid_step_lon = (self.lon_max - self.lon_min) / 10.0

            for lat in np.arange(self.lat_min, self.lat_max, grid_step_lat):
                for lon in np.arange(self.lon_min, self.lon_max, grid_step_lon):
                    u, v = inner_env._sample_wind_at(lat, lon, inner_env.alt_m)
                    if abs(u) > 1.0 or abs(v) > 1.0:
                        px, py = self._to_pixel(lat, lon)
                        spd_ms = math.sqrt(u**2 + v**2)
                        spd_kts = spd_ms / KTS_TO_M_S
                        arrow_len = min(40, max(10, spd_kts / 2.0))
                        dx = u / spd_ms * arrow_len
                        dy = -v / spd_ms * arrow_len
                        start_p = (px, py)
                        end_p = (px + dx, py + dy)
                        intensity = min(1.0, spd_kts / 100.0)
                        r = int(255 * intensity)
                        g = int(200 * (1 - intensity))
                        b = int(255 * (1 - intensity)) + 50
                        color = (min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b)))
                        pygame.draw.line(self.screen, color, start_p, end_p, 3)
                        pygame.draw.circle(self.screen, color, start_p, 3)

    def _draw_plane(self, lat, lon, heading, color=(255, 255, 0), label=None):
        plane_px = self._to_pixel(lat, lon)
        heading_rad = math.radians(heading)
        size = 12
        p1 = (plane_px[0] + math.sin(heading_rad) * size, plane_px[1] - math.cos(heading_rad) * size)
        p2 = (plane_px[0] + math.sin(heading_rad + 2.5) * size, plane_px[1] - math.cos(heading_rad + 2.5) * size)
        p3 = (plane_px[0] + math.sin(heading_rad - 2.5) * size, plane_px[1] - math.cos(heading_rad - 2.5) * size)
        pygame.draw.polygon(self.screen, color, [p1, p2, p3])
        
        if label:
            text = self.font.render(label, True, color)
            self.screen.blit(text, (plane_px[0] + 15, plane_px[1] - 10))

    def _draw_trajectory(self, trajectory, color=(255, 165, 0)):
        if len(trajectory) > 1:
            traj_points = [self._to_pixel(p[0], p[1]) for p in trajectory]
            pygame.draw.lines(self.screen, color, False, traj_points, 1)


class OpenAPVizWrapper(gym.Wrapper, BaseViz):
    """
    Gymnasium wrapper for 2D visualization of a single OpenAPNavEnv.
    """

    def __init__(self, env, width=800, height=800, padding=100):
        gym.Wrapper.__init__(self, env)
        BaseViz.__init__(self, width, height, padding)

        self.last_action_duration = 0.0
        self.total_duration_min = 0.0
        self.trajectory = []

    def step(self, action):
        self.trajectory.append((self.env.lat, self.env.lon))
        obs, reward, terminated, truncated, info = self.env.step(action)
        duration_min = info.get("duration", 5.0)
        self.last_action_duration = duration_min
        self.total_duration_min += duration_min
        self.last_segment_durations = info.get("segment_durations", [0.0, 0.0, 0.0])
        self.all_segment_durations = info.get("all_segment_durations", [])
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.total_duration_min = 0.0
        self.last_action_duration = 0.0
        self.last_segment_durations = [0.0, 0.0, 0.0]
        self.all_segment_durations = []
        self.trajectory = []
        obs, info = self.env.reset(**kwargs)
        self._calculate_bounds([self._get_env_data()])
        return obs, info

    def _get_env_data(self):
        return {
            'lat': self.env.lat,
            'lon': self.env.lon,
            'nominal_route': getattr(self.env, 'nominal_route', []),
            'trajectory': self.trajectory + [(self.env.lat, self.env.lon)]
        }

    def render(self):
        self._init_pygame()
        self._calculate_bounds([self._get_env_data()])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    self.show_segment_times = not self.show_segment_times

        self._draw_background()
        self._draw_nominal_route(getattr(self.env, 'nominal_route', []), self.env.current_waypoint_idx, self.all_segment_durations)
        self._draw_trajectory(self.trajectory + [(self.env.lat, self.env.lon)])
        self._draw_wind_field(self.env)
        self._draw_plane(self.env.lat, self.env.lon, self.env.heading_mag)

        # Info text
        info_lines = [
            f"Cap (HDG): {int(self.env.heading_mag)}Â°",
            f"Ecart Lat (XTE): {self.env._calculate_xte():.2f} NM",
            f"Dist Parcours (ATE): {self.env._calculate_atd():.2f} NM",
            f"DurÃ©e Totale: {self.total_duration_min:.1f} min",
            f"Segments (Last 3): {' / '.join([f'{d:.1f}' for d in self.last_segment_durations])}",
            f"Carburant: {int(self.env.current_fuel_kg)} kg",
            f"Vitesse TAS: {int(self.env.tas_ms / KTS_TO_M_S)} kts",
            f"Vitesse GS: {int(self.env.gs_ms / KTS_TO_M_S)} kts",
        ]
        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, 10 + i * 20))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


class MultiEnvViz(BaseViz):
    """
    Visualization for multiple environments at once.
    Not a wrapper, but takes a list of environments and their trackers.
    """

    def __init__(self, envs, labels=None, colors=None, width=1000, height=800):
        super().__init__(width, height)
        self.envs = envs
        self.labels = labels or [f"Env {i}" for i in range(len(envs))]
        self.colors = colors or self._get_default_colors(len(envs))
        
        # Trackers for each env
        self.trajectories = [[] for _ in envs]
        self.total_durations = [0.0 for _ in envs]
        self.last_durations = [0.0 for _ in envs]
        self.segment_durations_history = [[0.0, 0.0, 0.0] for _ in envs]
        self.all_segment_durations_history = [[] for _ in envs]
        self.xte_history = [[] for _ in envs]
        self.gs_history = [[] for _ in envs]

    def _get_default_colors(self, n):
        # Yellow, Cyan, Magenta, Green, Red...
        palette = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]
        return [palette[i % len(palette)] for i in range(n)]

    def update_env_info(self, idx, info):
        """Called after a step in one of the envs to update tracking info."""
        env = self.envs[idx]
        self.trajectories[idx].append((env.lat, env.lon))
        duration = info.get("duration", 5.0)
        self.total_durations[idx] += duration
        self.last_durations[idx] = duration
        self.segment_durations_history[idx] = info.get("segment_durations", [0.0, 0.0, 0.0])
        self.all_segment_durations_history[idx] = info.get("all_segment_durations", [])
        
        # Track history for averages
        self.xte_history[idx].append(abs(env._calculate_xte()))
        self.gs_history[idx].append(env.gs_ms / KTS_TO_M_S)

    def _get_envs_data(self):
        data = []
        for i, env in enumerate(self.envs):
            data.append({
                'lat': env.lat,
                'lon': env.lon,
                'nominal_route': getattr(env, 'nominal_route', []),
                'trajectory': self.trajectories[i] + [(env.lat, env.lon)]
            })
        return data

    def render(self):
        self._init_pygame("Multi-Env Benchmark Visualization")
        self._calculate_bounds(self._get_envs_data())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    self.show_segment_times = not self.show_segment_times

        self._draw_background()
        
        # 1. Draw nominal route (assume all envs share the same route for benchmarking)
        if self.envs:
            # Draw the static route first
            common_route = getattr(self.envs[0], 'nominal_route', [])
            self._draw_nominal_route(common_route, getattr(self.envs[0], 'current_waypoint_idx', 0))
            
            # Use small offsets for each plane's duration labels
            for i, env in enumerate(self.envs):
                # Offset each plane's labels vertically (15px per plane)
                offset_y = 5 + (i * 15)
                self._draw_nominal_route(
                    route=common_route, 
                    current_wp_idx=-1, 
                    durations=self.all_segment_durations_history[i],
                    label_color=self.colors[i],
                    label_offset=(5, offset_y),
                    draw_route=False
                )
            
            # 3. Calculate and draw delta if exactly 2 envs (Baseline vs Agent)
            if len(self.envs) == 2:
                ap_history = self.all_segment_durations_history[0]
                md_history = self.all_segment_durations_history[1]
                
                # Combine based on shortest history to avoid index errors
                min_len = min(len(ap_history), len(md_history))
                deltas = [md_history[j] - ap_history[j] for j in range(min_len)]
                
                offset_y = 5 + (2 * 15)
                self._draw_nominal_route(
                    route=common_route,
                    current_wp_idx=-1,
                    durations=deltas,
                    label_offset=(5, offset_y),
                    draw_route=False,
                    is_delta=True
                )

            self._draw_wind_field(self.envs[0])

        # 2. Draw each plane and its trajectory
        for i, env in enumerate(self.envs):
            traj = self.trajectories[i] + [(env.lat, env.lon)]
            self._draw_trajectory(traj, self.colors[i])
            self._draw_plane(env.lat, env.lon, env.heading_mag, self.colors[i], self.labels[i])

        # 3. Draw Info Table
        header_font = pygame.font.SysFont("Arial", 16, bold=True)
        headers = ["Env", "Dur(m)", "Segments (Last 3)", "XTE (min/avg/max)", "GS (min/avg/max)", "Fuel(kg)"]
        col_widths = [100, 60, 150, 165, 165, 80]
        start_x = 10
        start_y = 10
        
        # Draw background for info table
        pygame.draw.rect(self.screen, (30, 30, 40, 180), (start_x, start_y, sum(col_widths), 35 + len(self.envs) * 20))
        
        for j, h in enumerate(headers):
            txt = header_font.render(h, True, (200, 200, 200))
            self.screen.blit(txt, (start_x + sum(col_widths[:j]), start_y))

        for i, env in enumerate(self.envs):
            y = start_y + 30 + i * 20
            
            if self.xte_history[i]:
                min_xte = min(self.xte_history[i])
                avg_xte = sum(self.xte_history[i]) / len(self.xte_history[i])
                max_xte = max(self.xte_history[i])
                xte_str = f"{min_xte:.1f} / {avg_xte:.1f} / {max_xte:.1f}"
            else:
                xte_str = "- / - / -"

            if self.gs_history[i]:
                min_gs = min(self.gs_history[i])
                avg_gs = sum(self.gs_history[i]) / len(self.gs_history[i])
                max_gs = max(self.gs_history[i])
                gs_str = f"{int(min_gs)} / {int(avg_gs)} / {int(max_gs)}"
            else:
                gs_str = "- / - / -"
            
            durs = self.segment_durations_history[i]
            seg_str = f"{durs[0]:.1f} / {durs[1]:.1f} / {durs[2]:.1f}"
            
            values = [
                self.labels[i],
                f"{self.total_durations[i]:.1f}",
                seg_str,
                xte_str,
                gs_str,
                f"{int(env.current_fuel_kg)}"
            ]
            for j, val in enumerate(values):
                txt = self.font.render(val, True, self.colors[i])
                self.screen.blit(txt, (start_x + sum(col_widths[:j]), y))

        pygame.display.flip()
        self.clock.tick(60)


    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

