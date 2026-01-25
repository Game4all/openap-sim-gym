from dataclasses import dataclass, field
from pydantic import BaseModel
from typing import Dict, List, Optional


class WindStreamConfig(BaseModel):
    lat: float
    lon: float
    direction: float  # bearing in degrees
    width_nm: float   # Width standard deviation in NM
    max_speed_kts: float


class PlaneConfig(BaseModel):
    """Configuration specific to the aircraft."""
    aircraft_type: str

    # Initial State
    initial_lat: float = 48.0
    initial_lon: float = 2.0
    initial_alt_m: float = 10058.4  # ~33k ft
    initial_tas_ms: float = 231.5   # ~450 kts
    initial_fuel_kg: float = 15000.0

    # Initial heading. If None, it will be auto computed from route
    initial_heading_mag: Optional[float] = None


class EnvironmentConfig(BaseModel):
    """Configuration for the RL Environment simulation."""

    # Numbers of waypoints for which to expose information in the observation space
    lookahead_count: int = 3

    # Threshold distance the plane must be under to mark a waypoint as visited
    flyby_waypoint_dist: float = 20.0

    # Max number of steps before environment truncates.
    max_steps: int = 300
