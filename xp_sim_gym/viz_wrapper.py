import gymnasium as gym
import numpy as np
import pygame
import math
import time


class OpenAPVizWrapper(gym.Wrapper):
    """
    Un wrapper Gymnasium pour fournir une visualisation 2D pour OpenAPNavEnv en utilisant Pygame.
    """

    def __init__(self, env, width=800, height=800, padding=100):
        super().__init__(env)
        self.width = width
        self.height = height
        self.padding = padding

        self.screen = None
        self.clock = None
        self.font = None

        # Facteurs d'échelle (Limites de la carte)
        self.lon_min = None
        self.lon_max = None
        self.lat_min = None
        self.lat_max = None

        self.last_action_duration = 0.0
        self.total_duration_min = 0.0
        self.trajectory = []  # Liste des positions (lat, lon) parcourues

    def step(self, action):
        """Exécute une étape dans l'environnement et met à jour les données de visualisation."""
        # Enregistre la position avant l'étape pour le tracé de la trajectoire
        self.trajectory.append((self.env.lat, self.env.lon))

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get dynamic duration from info
        duration_min = info.get("duration", 5.0)
        self.last_action_duration = duration_min
        self.total_duration_min += duration_min

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Réinitialise l'environnement et les données de suivi de la visualisation."""
        self.total_duration_min = 0.0
        self.last_action_duration = 0.0
        self.trajectory = []
        obs, info = self.env.reset(**kwargs)
        self._calculate_bounds()
        return obs, info

    def _calculate_bounds(self):
        """Calcule les limites lat/lon pour s'assurer que tous les éléments importants sont visibles."""
        lats = [self.env.lat]
        lons = [self.env.lon]

        if hasattr(self.env, 'nominal_route') and self.env.nominal_route:
            lats.extend([wp['lat'] for wp in self.env.nominal_route])
            lons.extend([wp['lon'] for wp in self.env.nominal_route])

        if self.trajectory:
            lats.extend([p[0] for p in self.trajectory])
            lons.extend([p[1] for p in self.trajectory])

        self.lat_min, self.lat_max = min(lats), max(lats)
        self.lon_min, self.lon_max = min(lons), max(lons)

        # Définit un écart minimal pour éviter les divisions par zéro
        d_lat = max(self.lat_max - self.lat_min, 0.05)
        d_lon = max(self.lon_max - self.lon_min, 0.05)

        # Ajustement de l'aspect ratio (approximation)
        # 1 deg lat ~ 60 NM, 1 deg lon ~ 60 NM * cos(lat)
        aspect_ratio = self.width / self.height
        lat_center = (self.lat_min + self.lat_max) / 2
        lon_center = (self.lon_min + self.lon_max) / 2

        cos_lat = math.cos(math.radians(lat_center))

        if (d_lon * cos_lat) / d_lat > aspect_ratio:
            # Limité par la longitude
            d_lat = (d_lon * cos_lat) / aspect_ratio
        else:
            # Limité par la latitude
            d_lon = (d_lat * aspect_ratio) / cos_lat

        self.lat_min = lat_center - d_lat * 0.6
        self.lat_max = lat_center + d_lat * 0.6
        self.lon_min = lon_center - d_lon * 0.6
        self.lon_max = lon_center + d_lon * 0.6

    def _to_pixel(self, lat, lon):
        """Convertit les coordonnées lat/lon en pixels écran."""
        if self.lon_max == self.lon_min or self.lat_max == self.lat_min:
            return self.width // 2, self.height // 2

        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * self.width
        # Inverse l'axe Y (le point 0,0 de Pygame est en haut à gauche)
        y = self.height - (lat - self.lat_min) / \
            (self.lat_max - self.lat_min) * self.height
        return int(x), int(y)

    def render(self):
        """Affiche la fenêtre de visualisation et dessine les éléments de la simulation."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Simulation OpenAP - Visualisation 2D")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)

        # Mise à jour dynamique des limites pour suivre l'avion
        self._calculate_bounds()

        # Gestion des événements pour éviter que la fenêtre ne freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return

        self.screen.fill((20, 22, 28))  # Fond sombre

        # Dessin de la grille (indicatif)
        grid_color = (40, 45, 55)
        for lat in np.linspace(self.lat_min, self.lat_max, 10):
            p1 = self._to_pixel(lat, self.lon_min)
            p2 = self._to_pixel(lat, self.lon_max)
            pygame.draw.line(self.screen, grid_color, p1, p2, 1)
        for lon in np.linspace(self.lon_min, self.lon_max, 10):
            p1 = self._to_pixel(self.lat_min, lon)
            p2 = self._to_pixel(self.lat_max, lon)
            pygame.draw.line(self.screen, grid_color, p1, p2, 1)

        # 1. Dessin de la route nominale (Plan de vol)
        if hasattr(self.env, 'nominal_route') and self.env.nominal_route:
            points = [self._to_pixel(wp['lat'], wp['lon'])
                      for wp in self.env.nominal_route]
            if len(points) > 1:
                pygame.draw.lines(self.screen, (70, 70, 100), False, points, 2)

            for i, p in enumerate(points):
                # Le waypoint cible actuel est en vert, les autres en gris
                color = (0, 255, 127) if i == self.env.current_waypoint_idx else (
                    150, 150, 150)
                pygame.draw.rect(self.screen, color, (p[0]-3, p[1]-3, 6, 6))

        # 2. Dessin de la trajectoire réelle (Historique)
        if len(self.trajectory) > 1:
            traj_points = [self._to_pixel(p[0], p[1]) for p in self.trajectory]
            traj_points.append(self._to_pixel(self.env.lat, self.env.lon))
            pygame.draw.lines(self.screen, (255, 165, 0),
                              False, traj_points, 1)

        # 3. Dessin du champ de vent (Wind Field)
        # Visualise les flux de vent sur une grille
        if hasattr(self.env, 'wind_streams'):
            grid_step_lat = (self.lat_max - self.lat_min) / \
                10.0  # ~10x10 grid (Reduced from 15x15)
            grid_step_lon = (self.lon_max - self.lon_min) / 10.0

            for lat in np.arange(self.lat_min, self.lat_max, grid_step_lat):
                for lon in np.arange(self.lon_min, self.lon_max, grid_step_lon):
                    u, v = self.env._sample_wind_at(lat, lon, self.env.alt_m)

                    if abs(u) > 1.0 or abs(v) > 1.0:  # Ignore very light wind
                        px, py = self._to_pixel(lat, lon)

                        # Calculate mag and direction
                        spd_ms = math.sqrt(u**2 + v**2)
                        spd_kts = spd_ms / 0.514444

                        # Arrow length proportional to speed, clamped
                        # Scaled up for "bigger arrows" (Max 40px)
                        arrow_len = min(40, max(10, spd_kts / 2.0))

                        # Direction (Where wind goes TO)
                        # Pygame Y is down, so if V (North) is positive, it goes UP (-Y)
                        dx = u / spd_ms * arrow_len
                        dy = -v / spd_ms * arrow_len

                        start_p = (px, py)
                        end_p = (px + dx, py + dy)

                        # Color gradient based on speed
                        # Blue (calm) -> Red (Strong)
                        intensity = min(1.0, spd_kts / 100.0)
                        r = int(255 * intensity)
                        g = int(200 * (1 - intensity))
                        b = int(255 * (1 - intensity)) + 50

                        color = (
                            min(255, max(0, r)),
                            min(255, max(0, g)),
                            min(255, max(0, b))
                        )

                        pygame.draw.line(self.screen, color,
                                         start_p, end_p, 3)  # Thickness 3
                        # Medium dot at origin
                        pygame.draw.circle(self.screen, color, start_p, 3)

        # 4. Dessin de l'avion
        plane_px = self._to_pixel(self.env.lat, self.env.lon)
        # Triangle représentant l'avion
        heading_rad = math.radians(self.env.heading_mag)
        # Points : avant, arrière-gauche, arrière-droit
        size = 12
        p1 = (plane_px[0] + math.sin(heading_rad) * size,
              plane_px[1] - math.cos(heading_rad) * size)
        p2 = (plane_px[0] + math.sin(heading_rad + 2.5) * size,
              plane_px[1] - math.cos(heading_rad + 2.5) * size)
        p3 = (plane_px[0] + math.sin(heading_rad - 2.5) * size,
              plane_px[1] - math.cos(heading_rad - 2.5) * size)

        pygame.draw.polygon(self.screen, (255, 255, 0), [p1, p2, p3])

        # 5. Informations textuelles
        # Try to get ATE/XTE from env info if available, otherwise calculate
        xte = self.env._calculate_xte()
        ate = self.env._calculate_atd()

        info_lines = [
            f"Cap (HDG): {int(self.env.heading_mag)}°",
            f"Ecart Lat (XTE): {xte:.2f} NM",
            f"Dist Parcours (ATE): {ate:.2f} NM",
            f"Durée Totale: {self.total_duration_min:.1f} min",
            f"Durée Seg: {self.last_action_duration:.2f} min",
            f"Carburant: {int(self.env.current_fuel_kg)} kg",
            f"Vitesse TAS: {int(self.env.tas_ms / 0.514444)} kts",
            f"Vitesse GS: {int(self.env.gs_ms / 0.514444)} kts",
            f"Niveau (Stage): {self.env.stage}"
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
