from dataclasses import dataclass, field
from pydantic import BaseModel
from typing import Dict, List, Optional


class WindStreamConfig(BaseModel):
    lat: float
    lon: float
    direction: float  # bearing in degrees
    width_nm: float   # Width standard deviation in NM
    max_speed_kts: float


class PlaneEnvironmentConfig(BaseModel):
    aircraft_type: str
    lookahead_count: int = 3
    nominal_route: Optional[List[Dict]] = None

    # Initial Plane State

    initial_lat: float = 48.0
    initial_lon: float = 2.0
    initial_alt_m: float = 10058.4  # ~33k ft
    initial_tas_ms: float = 231.5  # ~450 kts
    initial_fuel_kg: float = 15000.0

    # If None, will be calculated from route

    initial_heading_mag: Optional[float] = None

    # Wind Config
    # Wind Config - Jet Stream / River Model
    # List of wind streams. If empty and randomize_wind is True, streams will be generated based on stage.
    wind_streams: List[WindStreamConfig] = field(default_factory=list)
    randomize_wind: bool = True

    # Threshold distance the plane must be under to mark a waypoint as visited
    flyby_waypoint_dist: float = 20.0

    # Max number of steps before environment truncates.
    # NOTE: Keep this high when benchmarking to prevent baseline from not completing.
    max_steps: int = 300
