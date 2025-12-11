"""Constants for the forestfire prediction project"""

from enum import Enum
from typing import List


class DataConstants:
    """Data-related constants"""

    # Grid dimensions
    GRID_HEIGHT = 278
    GRID_WIDTH = 400
    TOTAL_GRID_POINTS = GRID_HEIGHT * GRID_WIDTH  # 111,200

    # Missing value indicators
    MISSING_VALUE_GDAL = 32767.0  # GDAL no data value
    MISSING_VALUE_IDW = -9999.0  # IDW interpolation no data value
    MISSING_VALUES = [MISSING_VALUE_GDAL, MISSING_VALUE_IDW]

    # Fire location marker (used in rasterization)
    FIRE_LOCATION_MARKER = 624.0

    # Gangwon province boundaries (EPSG:4326)
    BOUNDARY_NORTH = 38.61370931
    BOUNDARY_EAST = 129.359995
    BOUNDARY_SOUTH = 37.03353708
    BOUNDARY_WEST = 127.0950376

    # Feature names
    CLIMATE_FEATURES = ["humidity", "rainfall", "temp", "windspeed"]
    SPATIAL_FEATURES = ["height", "slope", "ndvi", "landuse", "population_density"]
    ALL_FEATURES = CLIMATE_FEATURES + SPATIAL_FEATURES

    # Column name mappings (for different data sources)
    ASOS_COLUMN_MAP = {
        "일시": "time",
        "지점": "loc_info",
        "습도(%)": "humidity",
        "풍속(m/s)": "windspeed",
        "강수량(mm)": "rainfall",
        "기온(°C)": "temp",
    }

    FUTURE_WEATHER_COLUMN_MAP = {
        "REH": "humidity",  # 습도
        "RN1": "rainfall",  # 강수량
        "T1H": "temp",  # 기온
        "WSD": "windspeed",  # 풍속
    }


class PathConstants:
    """Path-related constants"""

    # Data directories
    RAW_DATA = "raw"
    PROCESSED_DATA = "processed"
    MODELING_DATA = "modeling"

    # Subdirectories
    CLIMATE_DATA = "climate_data"
    INTERPOLATE_CLIMATE = "interpolate_climate"
    IMAGE_DATA = "image_data"
    FIRE_LOC = "fire_loc"
    GEO_DATA = "geo_data"

    # File patterns
    NDVI_PATTERN = "NDVI/{year}{month:02d}.tif"
    HEIGHT_FILE = "Height_gw.tif"
    SLOPE_FILE = "Slope_gw.tif"
    LANDUSE_FILE = "Landuse_gw.tif"
    POPDEN_FILE = "population_density_gw.tif"


class ModelConstants:
    """Model-related constants"""

    # Random seeds
    RANDOM_SEED = 42

    # Data split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    # Class labels
    CLASS_FIRE = 1
    CLASS_NO_FIRE = 0


class APIConstants:
    """API-related constants"""

    # ASOS (Automated Surface Observing System) API
    ASOS_BASE_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"

    # VilageFcst (Village Forecast) API
    VILAGE_FCST_BASE_URL = (
        "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
    )

    # Request parameters
    MAX_RETRIES = 5
    RETRY_DELAY = 2  # seconds
    REQUEST_TIMEOUT = 30  # seconds


class SeasonConstants:
    """Season-related constants for random data generation"""

    SPRING_MONTHS = [2, 3, 4, 5]
    SUMMER_MONTHS = [6, 7, 8, 9, 10]  # Named 'else' in original code
    FALL_MONTHS = [11, 12]
    WINTER_MONTHS = [1]

    # Season weights for random sampling
    SEASON_WEIGHTS = {
        "spring": 0.6,
        "summer": 0.2,
        "fall": 0.1,
        "winter": 0.1,
    }


class NormalizationMethod(Enum):
    """Normalization methods"""

    STANDARD = "standard"  # StandardScaler (mean=0, std=1)
    MINMAX = "minmax"  # MinMaxScaler (0-1)
    ROBUST = "robust"  # RobustScaler (median, IQR)


class MissingValueStrategy(Enum):
    """Missing value handling strategies"""

    MASK = "mask"  # Create mask, keep values
    DROP = "drop"  # Drop samples with missing values
    MEAN = "mean"  # Fill with mean
    MEDIAN = "median"  # Fill with median
    ZERO = "zero"  # Fill with zero
