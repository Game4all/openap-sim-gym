
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from .config import PlaneEnvironmentConfig, WindStreamConfig
from .utils import GeoUtils


class RouteStageGenerator:
    def __init__(self, config: PlaneEnvironmentConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def generate(self, stage: int) -> Tuple[List[Dict], List[WindStreamConfig]]:
        """
        Generates a route and corresponding wind streams based on the difficulty stage.
        """

        route = self.generate_route(stage)

        # 2. Determine start params for wind generation
        if route:
            start_lat = route[0].get('lat', self.config.initial_lat)
            start_lon = route[0].get('lon', self.config.initial_lon)
        else:
            start_lat = self.config.initial_lat
            start_lon = self.config.initial_lon

        # 3. Generate Wind
        wind_streams = self.generate_wind(stage, route, start_lat, start_lon)

        return route, wind_streams

    def generate_route(self, stage: int) -> List[Dict]:
        """Génère un plan de vol aléatoire en fonction du stage de prétraining"""
        # Base location from config
        lat_0, lon_0 = self.config.initial_lat, self.config.initial_lon
        alt_0 = self.config.initial_alt_m

        route = []
        route.append({'lat': lat_0, 'lon': lon_0, 'alt': alt_0})

        if stage == 1:
            # Short straight route: 1 segment, ~50-100 NM
            dist_nm = np.random.uniform(50, 100)
            bearing = np.random.uniform(0, 360)

            # Rough calc for next point
            # 1 deg lat ~ 60 NM
            d_lat = dist_nm / 60.0 * math.cos(math.radians(bearing))
            d_lon = dist_nm / 60.0 * \
                math.sin(math.radians(bearing)) / math.cos(math.radians(lat_0))

            route.append(
                {'lat': lat_0 + d_lat, 'lon': lon_0 + d_lon, 'alt': alt_0})

        elif stage == 2:
            # Medium route: 2-3 segments, ~200 NM total
            num_segments = np.random.randint(2, 4)
            current_lat, current_lon = lat_0, lon_0
            global_bng = np.random.uniform(0, 360)
            max_deviation = 20

            for _ in range(num_segments):
                dist_nm = np.random.uniform(50, 80)
                # Deviation from global bearing
                current_bng = (
                    global_bng + np.random.uniform(-max_deviation, max_deviation)) % 360

                d_lat = dist_nm / 60.0 * math.cos(math.radians(current_bng))
                d_lon = dist_nm / 60.0 * \
                    math.sin(math.radians(current_bng)) / \
                    math.cos(math.radians(current_lat))

                current_lat += d_lat
                current_lon += d_lon
                route.append(
                    {'lat': current_lat, 'lon': current_lon, 'alt': alt_0})

        elif stage == 3:
            # Stage 3: Medium/Complex route: 4-6 segments, ~400-600 NM
            num_segments = np.random.randint(4, 7)
            current_lat, current_lon = lat_0, lon_0
            global_bng = np.random.uniform(0, 360)
            max_deviation = 40

            for _ in range(num_segments):
                dist_nm = np.random.uniform(60, 100)
                # Deviation from global bearing
                current_bng = (
                    global_bng + np.random.uniform(-max_deviation, max_deviation)) % 360

                d_lat = dist_nm / 60.0 * math.cos(math.radians(current_bng))
                d_lon = dist_nm / 60.0 * \
                    math.sin(math.radians(current_bng)) / \
                    math.cos(math.radians(current_lat))

                current_lat += d_lat
                current_lon += d_lon
                route.append(
                    {'lat': current_lat, 'lon': current_lon, 'alt': alt_0})

        elif stage == 4:
            # Stage 4: Long route: 7-10 segments, ~800-1200 NM
            num_segments = np.random.randint(7, 14)
            current_lat, current_lon = lat_0, lon_0
            global_bng = np.random.uniform(0, 360)
            max_deviation = 60

            for _ in range(num_segments):
                dist_nm = np.random.uniform(80, 150)
                # Deviation from global bearing
                current_bng = (
                    global_bng + np.random.uniform(-max_deviation, max_deviation)) % 360

                d_lat = dist_nm / 60.0 * math.cos(math.radians(current_bng))
                d_lon = dist_nm / 60.0 * \
                    math.sin(math.radians(current_bng)) / \
                    math.cos(math.radians(current_lat))

                current_lat += d_lat
                current_lon += d_lon
                route.append(
                    {'lat': current_lat, 'lon': current_lon, 'alt': alt_0})

        elif stage == 5:
            # Stage 5: Very long/Extreme route: 10-15 segments, ~1500+ NM
            num_segments = np.random.randint(13, 19)
            current_lat, current_lon = lat_0, lon_0
            global_bng = np.random.uniform(0, 360)
            max_deviation = 80

            for _ in range(num_segments):
                dist_nm = np.random.uniform(100, 200)
                # Deviation from global bearing
                current_bng = (
                    global_bng + np.random.uniform(-max_deviation, max_deviation)) % 360

                d_lat = dist_nm / 60.0 * math.cos(math.radians(current_bng))
                d_lon = dist_nm / 60.0 * \
                    math.sin(math.radians(current_bng)) / \
                    math.cos(math.radians(current_lat))

                current_lat += d_lat
                current_lon += d_lon
                route.append(
                    {'lat': current_lat, 'lon': current_lon, 'alt': alt_0})

        elif stage == 6:
            # Stage 6: BENCHMARK MODE: 20-30 segments, ~3000+ NM
            num_segments = np.random.randint(20, 36)
            current_lat, current_lon = lat_0, lon_0
            global_bng = np.random.uniform(0, 360)
            max_deviation = 80

            for _ in range(num_segments):
                dist_nm = np.random.uniform(100, 200)
                # Deviation from global bearing
                current_bng = (
                    global_bng + np.random.uniform(-max_deviation, max_deviation)) % 360

                d_lat = dist_nm / 60.0 * math.cos(math.radians(current_bng))
                d_lon = dist_nm / 60.0 * \
                    math.sin(math.radians(current_bng)) / \
                    math.cos(math.radians(current_lat))

                current_lat += d_lat
                current_lon += d_lon
                route.append(
                    {'lat': current_lat, 'lon': current_lon, 'alt': alt_0})

        return route

    def generate_wind(self, stage: int, route: List[Dict], start_lat: float, start_lon: float) -> List[WindStreamConfig]:
        """Initialize the wind streams."""
        wind_streams = []

        # Use config streams if provided
        if self.config.wind_streams:
            # Note: Returning config streams directly. Caller should copy if mutation is executed.
            return list(self.config.wind_streams)

        # Randomize based on Stage
        if stage == 1:
            return []  # No wind

        # Helper to get random route point center
        center_lat, center_lon = start_lat, start_lon
        if route and len(route) > 1:
            mid_idx = len(route) // 2
            center_lat, center_lon = route[mid_idx]['lat'], route[mid_idx]['lon']

        if stage == 2:
            # Single stream, moderate, aligned with general route direction
            # Find general bearing from start to end
            last = route[-1]
            gen_bearing = GeoUtils.bearing(
                start_lat, start_lon, last['lat'], last['lon'])

            wind_streams.append(WindStreamConfig(
                lat=center_lat, lon=center_lon,
                direction=gen_bearing + np.random.uniform(-20, 20),
                width_nm=np.random.uniform(80, 150),
                max_speed_kts=np.random.uniform(20, 50)
            ))

        elif stage == 3:
            # Stronger stream, possibly misalignment
            last = route[-1]
            gen_bearing = GeoUtils.bearing(
                start_lat, start_lon, last['lat'], last['lon'])

            # Offset the stream center so we might enter/exit it
            offset_dist = np.random.uniform(-30, 30)  # NM
            # Simple lat offset approx
            center_lat += (offset_dist / 60.0)

            wind_streams.append(WindStreamConfig(
                lat=center_lat, lon=center_lon,
                direction=gen_bearing + np.random.uniform(-45, 45),
                width_nm=np.random.uniform(30, 80),  # Narrower = more gradient
                max_speed_kts=np.random.uniform(50, 90)
            ))

        elif stage == 4:
            # Two streams
            # 1. Main stream
            last = route[-1]
            gen_bearing = GeoUtils.bearing(
                start_lat, start_lon, last['lat'], last['lon'])

            wind_streams.append(WindStreamConfig(
                lat=center_lat, lon=center_lon,
                direction=gen_bearing + np.random.uniform(-30, 30),
                width_nm=np.random.uniform(50, 90),
                max_speed_kts=np.random.uniform(40, 80)
            ))

            # 2. Crossing/Interfering stream
            wind_streams.append(WindStreamConfig(
                lat=center_lat + np.random.uniform(-1, 1),
                lon=center_lon + np.random.uniform(-1, 1),
                direction=np.random.uniform(0, 360),
                width_nm=np.random.uniform(50, 90),
                max_speed_kts=np.random.uniform(30, 60)
            ))

        elif stage >= 5:
            # Chaos: 3 active streams
            for _ in range(3):
                wind_streams.append(WindStreamConfig(
                    lat=center_lat + np.random.uniform(-2, 2),
                    lon=center_lon + np.random.uniform(-2, 2),
                    direction=np.random.uniform(0, 360),
                    width_nm=np.random.uniform(40, 120),
                    max_speed_kts=np.random.uniform(40, 120)
                ))

        elif stage >= 6:
            # benchmark mode
            for _ in range(6):
                wind_streams.append(WindStreamConfig(
                    lat=center_lat + np.random.uniform(-2, 2),
                    lon=center_lon + np.random.uniform(-2, 2),
                    direction=np.random.uniform(0, 360),
                    width_nm=np.random.uniform(40, 120),
                    max_speed_kts=np.random.uniform(40, 120)
                ))

        return wind_streams
