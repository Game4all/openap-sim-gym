
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from .config import PlaneConfig, WindStreamConfig
from .utils import GeoUtils


class RouteStageGenerator:
    """
    Generates flight plans and corresponding winds based on the pre-training stage.
    """
    def __init__(self, config: PlaneConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def generate(self, stage: int) -> Tuple[List[Dict], List[WindStreamConfig]]:
        """
        Generates a flight plan and corresponding winds based on the pre-training stage.
        """

        route = self.generate_route(stage)

        if route:
            start_lat = route[0].get('lat', self.config.initial_lat)
            start_lon = route[0].get('lon', self.config.initial_lon)
        else:
            start_lat = self.config.initial_lat
            start_lon = self.config.initial_lon

        wind_streams = self.generate_wind(stage, route, start_lat, start_lon)

        return route, wind_streams

    def generate_route(self, stage: int) -> List[Dict]:
        """Generates a random flight plan based on the pre-training stage."""
        lat_0, lon_0 = self.config.initial_lat, self.config.initial_lon
        alt_0 = self.config.initial_alt_m

        route = []
        route.append({'lat': lat_0, 'lon': lon_0, 'alt': alt_0})

        if stage == 1:
            # Stage 1: Control & Geometry
            # 1 straight segment, Total distance: 50–100 NM, No heading changes
            dist_nm = np.random.uniform(50, 100)
            bearing = np.random.uniform(0, 360)

            # 1 deg lat ~ 60 NM
            d_lat = dist_nm / 60.0 * math.cos(math.radians(bearing))
            d_lon = dist_nm / 60.0 * \
                math.sin(math.radians(bearing)) / math.cos(math.radians(lat_0))

            route.append(
                {'lat': lat_0 + d_lat, 'lon': lon_0 + d_lon, 'alt': alt_0})

        elif stage == 2:
            # Stage 2: Basic Navigation
            # 2–3 segments, Mild bearing changes (≤ ±20°), Total distance: ~150–250 NM
            num_segments = np.random.randint(2, 4)
            total_dist_nm = np.random.uniform(150, 250)
            seg_dist_nm = total_dist_nm / num_segments
            
            current_lat, current_lon = lat_0, lon_0
            global_bng = np.random.uniform(0, 360)
            max_deviation = 20

            for _ in range(num_segments):
                # Deviation from global bearing
                current_bng = (
                    global_bng + np.random.uniform(-max_deviation, max_deviation)) % 360

                d_lat = seg_dist_nm / 60.0 * math.cos(math.radians(current_bng))
                d_lon = seg_dist_nm / 60.0 * \
                    math.sin(math.radians(current_bng)) / \
                    math.cos(math.radians(current_lat))

                current_lat += d_lat
                current_lon += d_lon
                route.append(
                    {'lat': current_lat, 'lon': current_lon, 'alt': alt_0})

        elif stage == 3:
            # Stage 3: Wind Exploitation
            # 1–2 long straight segments, Segment length: 150–300 NM, No sharp turns
            num_segments = np.random.randint(1, 3)
            current_lat, current_lon = lat_0, lon_0
            global_bng = np.random.uniform(0, 360)
            
            for _ in range(num_segments):
                dist_nm = np.random.uniform(150, 300)
                # Very small deviation to keep it "straight"
                current_bng = (global_bng + np.random.uniform(-5, 5)) % 360

                d_lat = dist_nm / 60.0 * math.cos(math.radians(current_bng))
                d_lon = dist_nm / 60.0 * \
                    math.sin(math.radians(current_bng)) / \
                    math.cos(math.radians(current_lat))

                current_lat += d_lat
                current_lon += d_lon
                route.append(
                    {'lat': current_lat, 'lon': current_lon, 'alt': alt_0})

        elif stage == 4:
            # Stage 4: Tradeoff Navigation
            # 3–5 segments, Moderate bearing changes (≤ ±40°), Total distance: ~300–600 NM
            num_segments = np.random.randint(3, 6)
            total_dist_nm = np.random.uniform(300, 600)
            seg_dist_nm = total_dist_nm / num_segments
            
            current_lat, current_lon = lat_0, lon_0
            global_bng = np.random.uniform(0, 360)
            max_deviation = 40

            for _ in range(num_segments):
                current_bng = (
                    global_bng + np.random.uniform(-max_deviation, max_deviation)) % 360

                d_lat = seg_dist_nm / 60.0 * math.cos(math.radians(current_bng))
                d_lon = seg_dist_nm / 60.0 * \
                    math.sin(math.radians(current_bng)) / \
                    math.cos(math.radians(current_lat))

                current_lat += d_lat
                current_lon += d_lon
                route.append(
                    {'lat': current_lat, 'lon': current_lon, 'alt': alt_0})

        elif stage >= 5:
            # Stage 5: Long-Horizon Navigation
            # 6–10 segments, Bearing changes up to ±60°, Total distance: ~800–1200 NM
            num_segments = np.random.randint(6, 11)
            total_dist_nm = np.random.uniform(800, 1200)
            seg_dist_nm = total_dist_nm / num_segments
            
            current_lat, current_lon = lat_0, lon_0
            global_bng = np.random.uniform(0, 360)
            max_deviation = 60

            for _ in range(num_segments):
                current_bng = (
                    global_bng + np.random.uniform(-max_deviation, max_deviation)) % 360

                d_lat = seg_dist_nm / 60.0 * math.cos(math.radians(current_bng))
                d_lon = seg_dist_nm / 60.0 * \
                    math.sin(math.radians(current_bng)) / \
                    math.cos(math.radians(current_lat))

                current_lat += d_lat
                current_lon += d_lon
                route.append(
                    {'lat': current_lat, 'lon': current_lon, 'alt': alt_0})

        return route
    def generate_wind(self, stage: int, route: List[Dict], start_lat: float, start_lon: float) -> List[WindStreamConfig]:
        """Generates winds corresponding to the pre-training stage."""
        wind_streams = []

        if stage == 1:
            return []  # No wind

        # Helper to get random route point center
        mid_idx = len(route) // 2
        center_lat, center_lon = route[mid_idx]['lat'], route[mid_idx]['lon']
        
        # General route bearing
        last = route[-1]
        gen_bearing = GeoUtils.bearing(start_lat, start_lon, last['lat'], last['lon'])

        if stage == 2:
            # Optional, weak wind aligned with route
            if np.random.random() > 0.5:
                wind_streams.append(WindStreamConfig(
                    lat=center_lat, lon=center_lon,
                    direction=gen_bearing + np.random.uniform(-10, 10),
                    width_nm=np.random.uniform(100, 200),
                    max_speed_kts=np.random.uniform(10, 30)
                ))

        elif stage == 3:
            # Parallel jet stream offset 5–9 NM from track
            # Narrow, strong wind corridor
            
            # Choose an offset direction perpendicular to the general bearing
            offset_bng = (gen_bearing + 90) if np.random.random() > 0.5 else (gen_bearing - 90)
            offset_dist_nm = np.random.uniform(5, 9)
            
            # Calculate offset center
            d_lat_off = offset_dist_nm / 60.0 * math.cos(math.radians(offset_bng))
            d_lon_off = offset_dist_nm / 60.0 * \
                math.sin(math.radians(offset_bng)) / math.cos(math.radians(center_lat))
            
            wind_streams.append(WindStreamConfig(
                lat=center_lat + d_lat_off,
                lon=center_lon + d_lon_off,
                direction=gen_bearing, # Perfectly aligned with track but offset
                width_nm=np.random.uniform(10, 20), # Narrow corridor
                max_speed_kts=np.random.uniform(60, 100)
            ))

        elif stage == 4:
            # Mix of aligned, offset-parallel, and crossing winds
            # 1. Aligned
            wind_streams.append(WindStreamConfig(
                lat=center_lat, lon=center_lon,
                direction=gen_bearing,
                width_nm=np.random.uniform(40, 80),
                max_speed_kts=np.random.uniform(30, 60)
            ))
            # 2. Offset-parallel
            offset_bng = (gen_bearing + 90) if np.random.random() > 0.5 else (gen_bearing - 90)
            offset_dist_nm = np.random.uniform(15, 30)
            d_lat_off = offset_dist_nm / 60.0 * math.cos(math.radians(offset_bng))
            d_lon_off = offset_dist_nm / 60.0 * \
                math.sin(math.radians(offset_bng)) / math.cos(math.radians(center_lat))
            
            wind_streams.append(WindStreamConfig(
                lat=center_lat + d_lat_off,
                lon=center_lon + d_lon_off,
                direction=gen_bearing,
                width_nm=np.random.uniform(20, 40),
                max_speed_kts=np.random.uniform(40, 80)
            ))
            # 3. Crossing
            wind_streams.append(WindStreamConfig(
                lat=center_lat + np.random.uniform(-0.5, 0.5),
                lon=center_lon + np.random.uniform(-0.5, 0.5),
                direction=(gen_bearing + 90) % 360,
                width_nm=np.random.uniform(50, 100),
                max_speed_kts=np.random.uniform(20, 50)
            ))

        elif stage >= 5:
            # Multiple wind streams with varying alignment
            num_streams = np.random.randint(4, 7)
            for _ in range(num_streams):
                wind_streams.append(WindStreamConfig(
                    lat=center_lat + np.random.uniform(-1.5, 1.5),
                    lon=center_lon + np.random.uniform(-1.5, 1.5),
                    direction=np.random.uniform(0, 360),
                    width_nm=np.random.uniform(30, 150),
                    max_speed_kts=np.random.uniform(40, 100)
                ))

        return wind_streams


class BenchmarkRouteGenerator:
    """
    Generates flight plans and corresponding winds for benchmarking.
    """ 
    def __init__(self, config: PlaneConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def generate(self) -> Tuple[List[Dict], List[WindStreamConfig]]:
        """
        Generates a flight plan and corresponding winds for benchmarking.
        """
        route = self.generate_route()

        if route:
            start_lat = route[0].get('lat', self.config.initial_lat)
            start_lon = route[0].get('lon', self.config.initial_lon)
        else:
            start_lat = self.config.initial_lat
            start_lon = self.config.initial_lon

        wind_streams = self.generate_wind(route, start_lat, start_lon)

        return route, wind_streams

    def generate_route(self) -> List[Dict]:
        """Generates a long flight plan (equivalent to Stage 5)."""
        lat_0, lon_0 = self.config.initial_lat, self.config.initial_lon
        alt_0 = self.config.initial_alt_m

        route = []
        route.append({'lat': lat_0, 'lon': lon_0, 'alt': alt_0})

        # Stage 5: Long-Horizon Navigation
        num_segments = np.random.randint(14, 25)
        total_dist_nm = np.random.uniform(1200, 2000)
        seg_dist_nm = total_dist_nm / num_segments
        
        current_lat, current_lon = lat_0, lon_0
        global_bng = np.random.uniform(0, 360)
        max_deviation = 60

        for _ in range(num_segments):
            current_bng = (
                global_bng + np.random.uniform(-max_deviation, max_deviation)) % 360

            d_lat = seg_dist_nm / 60.0 * math.cos(math.radians(current_bng))
            d_lon = seg_dist_nm / 60.0 * \
                math.sin(math.radians(current_bng)) / \
                math.cos(math.radians(current_lat))

            current_lat += d_lat
            current_lon += d_lon
            route.append(
                {'lat': current_lat, 'lon': current_lon, 'alt': alt_0})

        return route

    def generate_wind(self, route: List[Dict], start_lat: float, start_lon: float) -> List[WindStreamConfig]:
        """Generates winds corresponding to the benchmark route."""
        wind_streams = []
        num_streams = np.random.randint(4, 8)
        
        # Randomly determine headwind/tailwind counts
        max_special = num_streams // 2
        num_headwinds = np.random.randint(0, max_special + 1)
        num_tailwinds = np.random.randint(1, max_special + 1)
        num_random = max(0, num_streams - num_headwinds - num_tailwinds)

        # Helper to place wind alongside a segment
        def place_aligned_wind(is_tailwind: bool):
            if len(route) < 2: return
            
            # Pick a random segment index
            seg_idx = np.random.randint(0, len(route) - 1)
            p1 = route[seg_idx]
            p2 = route[seg_idx + 1]
            
            # Segment bearing
            seg_bng = GeoUtils.bearing(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
            
            # Segment midpoint
            mid_lat = (p1['lat'] + p2['lat']) / 2.0
            mid_lon = (p1['lon'] + p2['lon']) / 2.0
            
            # Perpendicular offset
            offset_bng = (seg_bng + 90) if np.random.random() > 0.5 else (seg_bng - 90)
            offset_dist_nm = np.random.uniform(10, 30)
            
            d_lat_off = offset_dist_nm / 60.0 * math.cos(math.radians(offset_bng))
            d_lon_off = offset_dist_nm / 60.0 * \
                math.sin(math.radians(offset_bng)) / math.cos(math.radians(mid_lat))
            
            direction = seg_bng if is_tailwind else (seg_bng + 180) % 360
            
            wind_streams.append(WindStreamConfig(
                lat=mid_lat + d_lat_off,
                lon=mid_lon + d_lon_off,
                direction=direction,
                width_nm=np.random.uniform(20, 40),
                max_speed_kts=np.random.uniform(60, 120)
            ))

        # Generate headwinds
        for _ in range(num_headwinds):
            place_aligned_wind(is_tailwind=False)

        # Generate tailwinds
        for _ in range(num_tailwinds):
            place_aligned_wind(is_tailwind=True)

        # Generate random streams for background "noise"
        # Use center of the route area
        avg_lat = sum(p['lat'] for p in route) / len(route)
        avg_lon = sum(p['lon'] for p in route) / len(route)
        
        for _ in range(num_random):
            wind_streams.append(WindStreamConfig(
                lat=avg_lat + np.random.uniform(-1.5, 1.5),
                lon=avg_lon + np.random.uniform(-1.5, 1.5),
                direction=np.random.uniform(0, 360),
                width_nm=np.random.uniform(50, 150),
                max_speed_kts=np.random.uniform(30, 70)
            ))

        return wind_streams

        return wind_streams
