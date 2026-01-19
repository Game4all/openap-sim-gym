from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PlaneEnvironmentConfig:
    aircraft_type: str
    lookahead_count: int = 3
    nominal_route: Optional[List[Dict]] = None

    # Initial State
    initial_lat: float = 48.0
    initial_lon: float = 2.0
    initial_alt_m: float = 10058.4  # ~33k ft
    initial_tas_ms: float = 231.5  # ~450 kts
    initial_fuel_kg: float = 15000.0

    # If None, will be calculated from route
    initial_heading_mag: Optional[float] = None

    # Wind Config
    wind_u: Optional[float] = None  # Flow component East
    wind_v: Optional[float] = None  # Flow component North
    randomize_wind: bool = True
