import math


class GeoUtils:
    R_EARTH_NM = 3440.06479  # Earth radius in nautical miles

    @staticmethod
    def haversine_dist(lat1, lon1, lat2, lon2):
        """Calculates the distance in NM between two points."""
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi/2)**2 + math.cos(phi1) * \
            math.cos(phi2) * math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return GeoUtils.R_EARTH_NM * c

    @staticmethod
    def bearing(lat1, lon1, lat2, lon2):
        """Calculates the initial bearing in degrees (0-360) from point 1 to point 2."""
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dlambda = math.radians(lon2 - lon1)

        y = math.sin(dlambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * \
            math.cos(phi2) * math.cos(dlambda)
        theta = math.atan2(y, x)
        return (math.degrees(theta) + 360) % 360

    @staticmethod
    def cross_track_error(lat_pos, lon_pos, lat_start, lon_start, lat_end, lon_end):
        """Calculates Cross Track Error (XTE) in NM. Positive = right of track, Negative = left of track."""
        dist_13 = GeoUtils.haversine_dist(
            lat_start, lon_start, lat_pos, lon_pos)
        brg_13 = GeoUtils.bearing(lat_start, lon_start, lat_pos, lon_pos)
        brg_12 = GeoUtils.bearing(lat_start, lon_start, lat_end, lon_end)

        rel_brg = math.radians(brg_13 - brg_12)
        return dist_13 * math.sin(rel_brg)

    @staticmethod
    def along_track_distance(lat_pos, lon_pos, lat_start, lon_start, lat_end, lon_end):
        """Calculates Along Track Distance (ATD) in NM from the start to the position's projection on the segment."""
        dist_13 = GeoUtils.haversine_dist(
            lat_start, lon_start, lat_pos, lon_pos)
        brg_13 = GeoUtils.bearing(lat_start, lon_start, lat_pos, lon_pos)
        brg_12 = GeoUtils.bearing(lat_start, lon_start, lat_end, lon_end)

        rel_brg = math.radians(brg_13 - brg_12)
        return dist_13 * math.cos(rel_brg)
