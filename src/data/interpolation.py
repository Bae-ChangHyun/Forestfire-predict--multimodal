"""Spatial interpolation utilities using IDW (Inverse Distance Weighting)"""

import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from pathlib import Path
from pyidw import idw
from tqdm import tqdm
from typing import List, Optional, Dict
import logging

from ..utils.constants import DataConstants

logger = logging.getLogger(__name__)


class IDWInterpolator:
    """IDW (Inverse Distance Weighting) interpolation for climate data"""

    def __init__(
        self,
        extent_shapefile: str,
        power: float = 2.0,
        search_radius: float = 8.0,
        output_resolution: int = 400,
    ):
        """
        Initialize IDW interpolator

        Args:
            extent_shapefile: Path to boundary shapefile (e.g., Gangwon province)
            power: Distance power parameter
            search_radius: Search radius for neighboring points
            output_resolution: Output raster resolution
        """
        self.extent_shapefile = extent_shapefile
        self.power = power
        self.search_radius = search_radius
        self.output_resolution = output_resolution

    def interpolate(
        self, input_shapefile: str, feature_name: str, output_path: str
    ) -> pd.DataFrame:
        """
        Perform IDW interpolation

        Args:
            input_shapefile: Path to input point shapefile
            feature_name: Name of the feature column to interpolate
            output_path: Path to save output raster (.tif)

        Returns:
            DataFrame with interpolated values
        """
        # Perform IDW interpolation
        idw.idw_interpolation(
            input_point_shapefile=input_shapefile,
            extent_shapefile=self.extent_shapefile,
            column_name=feature_name,
            power=self.power,
            search_radious=self.search_radius,  # Note: typo in library parameter name
            output_resolution=self.output_resolution,
        )

        # Default output name from pyidw library
        temp_output = input_shapefile.replace(".shp", "_idw.tif")

        # Read interpolated raster
        with rasterio.open(temp_output) as src:
            data = src.read(1)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Move to desired location if different
        if output_path != temp_output:
            os.rename(temp_output, output_path)

        return df

    def interpolate_multiple_features(
        self,
        input_shapefile: str,
        features: List[str],
        output_dir: str,
        base_name: str = "data",
    ) -> Dict[str, pd.DataFrame]:
        """
        Interpolate multiple features

        Args:
            input_shapefile: Path to input point shapefile
            features: List of feature names to interpolate
            output_dir: Directory to save output files
            base_name: Base name for output files

        Returns:
            Dictionary mapping feature names to interpolated DataFrames
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        for feature in tqdm(features, desc=f"Interpolating {base_name}"):
            output_path = Path(output_dir) / f"{base_name}_{feature}.tif"
            df = self.interpolate(input_shapefile, feature, str(output_path))
            results[feature] = df

        return results


def interpolate_climate_batch(
    data_dir: str,
    climate_data: pd.DataFrame,
    features: List[str],
    extent_shapefile: str,
    output_dir: str,
    start_idx: int = 0,
) -> None:
    """
    Batch interpolate climate data for multiple time points

    Args:
        data_dir: Directory to save temporary shapefiles
        climate_data: DataFrame with climate data (must have 'num' column for grouping)
        features: List of features to interpolate
        extent_shapefile: Path to boundary shapefile
        output_dir: Directory to save interpolated data
        start_idx: Starting index (for resuming)
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create feature subdirectories
    for feature in features:
        os.makedirs(Path(output_dir) / feature, exist_ok=True)

    # Group by time point
    grouped = climate_data.groupby("num")
    total_groups = climate_data["num"].nunique()

    # Check for existing data
    first_feature_dir = Path(output_dir) / features[0]
    existing_count = len(list(first_feature_dir.glob("*.csv")))

    if existing_count == total_groups:
        logger.info("All interpolated data already exists. Skipping.")
        return
    elif existing_count > start_idx:
        logger.info(f"Resuming from index {existing_count}")
        start_idx = existing_count

    # Initialize interpolator
    interpolator = IDWInterpolator(extent_shapefile=extent_shapefile)

    # Process each time point
    for num, group in tqdm(grouped, desc="Interpolating climate data"):
        if num < start_idx:
            continue

        # Check if data exists
        if len(group) == 0:
            logger.warning(f"No data for index {num}")
            continue

        # Drop missing values
        group_clean = group.dropna()
        if len(group_clean) == 0:
            logger.warning(f"All data missing for index {num}")
            continue

        # Create temporary shapefile
        temp_shp = Path(data_dir) / f"temp_{num}.shp"
        gdf = gpd.GeoDataFrame(
            group_clean, geometry=gpd.points_from_xy(group_clean["lon"], group_clean["lat"])
        )
        gdf.crs = "EPSG:4326"
        gdf.to_file(temp_shp)

        # Interpolate each feature
        for feature in features:
            output_csv = Path(output_dir) / feature / f"data{num}_idw.csv"

            if output_csv.exists():
                continue

            try:
                temp_tif = str(temp_shp).replace(".shp", f"_{feature}_idw.tif")
                df_interp = interpolator.interpolate(str(temp_shp), feature, temp_tif)
                df_interp.to_csv(output_csv, encoding="utf-8")

                # Clean up temp tif
                if os.path.exists(temp_tif):
                    os.remove(temp_tif)

            except Exception as e:
                logger.error(f"Failed to interpolate {feature} for index {num}: {e}")

        # Clean up temporary shapefile
        for ext in [".shp", ".shx", ".dbf", ".cpg", ".prj"]:
            temp_file = str(temp_shp).replace(".shp", ext)
            if os.path.exists(temp_file):
                os.remove(temp_file)

    logger.info("Batch interpolation completed")


def extract_fire_location_values(
    interpolated_dir: str, fire_locations: pd.DataFrame, features: List[str]
) -> pd.DataFrame:
    """
    Extract interpolated values at fire locations

    Args:
        interpolated_dir: Directory with interpolated CSV files
        fire_locations: DataFrame with fire locations (must have fire location raster data)
        features: List of features to extract

    Returns:
        DataFrame with extracted values
    """
    results = []

    for idx, row in tqdm(fire_locations.iterrows(), total=len(fire_locations), desc="Extracting values"):
        # Load fire location raster to find the exact location
        fire_raster_path = Path(interpolated_dir).parent / "fire_loc" / f"fire_data{idx}.csv"

        if not fire_raster_path.exists():
            logger.warning(f"Fire location file not found: {fire_raster_path}")
            continue

        fire_data = pd.read_csv(fire_raster_path, index_col=0)

        # Find fire location (marked with FIRE_LOCATION_MARKER)
        fire_mask = np.isclose(fire_data.values, DataConstants.FIRE_LOCATION_MARKER)

        if not fire_mask.any():
            logger.warning(f"Fire location marker not found for index {idx}")
            continue

        # Get row and column indices
        fire_row, fire_col = np.where(fire_mask)
        r, c = fire_row[0], fire_col[0]

        # Extract values from interpolated data
        values = {}
        for feature in features:
            interp_file = Path(interpolated_dir) / feature / f"data{idx}_idw.csv"

            if not interp_file.exists():
                logger.warning(f"Interpolated file not found: {interp_file}")
                values[feature] = np.nan
                continue

            interp_data = pd.read_csv(interp_file, index_col=0)
            values[feature] = interp_data.iloc[r, c]

        results.append(values)

    return pd.DataFrame(results)
