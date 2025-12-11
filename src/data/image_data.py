"""Image data processing module for spatial features"""

import os
import cv2
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm import tqdm
from itertools import product
from typing import List, Tuple, Optional
import logging

from osgeo import gdal
from pyproj import Transformer
import natsort

from ..utils.constants import DataConstants, PathConstants

logger = logging.getLogger(__name__)


class ImageCropper:
    """Crop raster images to specific locations and sizes"""

    def __init__(self, reference_raster: str):
        """
        Initialize image cropper

        Args:
            reference_raster: Path to reference raster for boundary and resolution
        """
        self.reference_raster = reference_raster

        # Get reference image properties
        with rasterio.open(reference_raster) as rds:
            self.width = rds.width
            self.height = rds.height
            self.bounds = rds.bounds
            self.left, self.bottom, self.right, self.top = rds.bounds

        # Calculate resolution
        self.resolution_x = (self.right - self.left) / self.width
        self.resolution_y = (self.top - self.bottom) / self.height

    def crop_image(
        self,
        input_raster: str,
        output_path: str,
        latitude: float,
        longitude: float,
        crop_size: int = 1,
    ) -> None:
        """
        Crop image centered at given coordinates

        Args:
            input_raster: Path to input raster file
            output_path: Path to save cropped image
            latitude: Latitude of center point
            longitude: Longitude of center point
            crop_size: Half-size of crop window (in pixels)
        """
        # Transform coordinates (identity transform for EPSG:4326)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:4326")
        lon_transformed, lat_transformed = transformer.transform(longitude, latitude)

        # Calculate bounding box
        left_box = lat_transformed - (self.resolution_x * crop_size)
        top_box = lon_transformed + (self.resolution_y * crop_size)
        right_box = lat_transformed + (self.resolution_x * crop_size)
        bottom_box = lon_transformed - (self.resolution_y * crop_size)

        window = (left_box, top_box, right_box, bottom_box)

        # Crop using GDAL
        gdal.Translate(output_path, input_raster, projWin=window)


class NDVICropper(ImageCropper):
    """Specialized cropper for NDVI data (varies by year and month)"""

    def __init__(self, reference_raster: str, ndvi_data_dir: str):
        """
        Initialize NDVI cropper

        Args:
            reference_raster: Path to reference raster
            ndvi_data_dir: Directory containing NDVI .tif files
        """
        super().__init__(reference_raster)
        self.ndvi_data_dir = Path(ndvi_data_dir)

    def get_ndvi_file(self, date_str: str) -> str:
        """
        Get NDVI file path for given date

        Args:
            date_str: Date string in format 'YYYYMMDD'

        Returns:
            Path to NDVI file
        """
        year = date_str[:4]
        month = int(date_str[4:6])

        # Map month to season
        if 3 <= month <= 5:
            season_month = 4  # Spring
        elif 6 <= month <= 9:
            season_month = 8  # Summer
        elif 10 <= month <= 11:
            season_month = 11  # Fall
        else:
            season_month = 1  # Winter

        filename = f"{year}{season_month:02d}.tif"
        return str(self.ndvi_data_dir / filename)

    def crop_ndvi(
        self,
        output_path: str,
        date_str: str,
        latitude: float,
        longitude: float,
        crop_size: int = 1,
    ) -> None:
        """
        Crop NDVI image for given date and location

        Args:
            output_path: Path to save cropped image
            date_str: Date string in format 'YYYYMMDD'
            latitude: Latitude
            longitude: Longitude
            crop_size: Crop window half-size
        """
        ndvi_file = self.get_ndvi_file(date_str)

        if not os.path.exists(ndvi_file):
            raise FileNotFoundError(f"NDVI file not found: {ndvi_file}")

        self.crop_image(ndvi_file, output_path, latitude, longitude, crop_size)


def crop_images_for_training(
    data_df: pd.DataFrame,
    feature_name: str,
    output_dir: str,
    reference_raster: str,
    input_raster: Optional[str] = None,
    crop_size: int = 1,
    is_ndvi: bool = False,
    ndvi_data_dir: Optional[str] = None,
) -> None:
    """
    Crop images for all samples in training data

    Args:
        data_df: DataFrame with sample information (must have 'date', 'lon', 'lat' columns)
        feature_name: Name of the feature (e.g., 'Height', 'Slope')
        output_dir: Directory to save cropped images
        reference_raster: Path to reference raster
        input_raster: Path to input raster (not used for NDVI)
        crop_size: Crop window half-size
        is_ndvi: Whether this is NDVI data
        ndvi_data_dir: Directory with NDVI files (for NDVI only)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if already processed
    existing_files = natsort.natsorted(list(Path(output_dir).glob("*.tif")))
    if len(existing_files) == len(data_df):
        logger.info(f"{feature_name} training crop images already exist")
        return

    logger.info(f"Creating {feature_name} training crop images")

    if is_ndvi:
        if ndvi_data_dir is None:
            raise ValueError("ndvi_data_dir must be provided for NDVI")
        cropper = NDVICropper(reference_raster, ndvi_data_dir)
    else:
        if input_raster is None:
            raise ValueError("input_raster must be provided for non-NDVI features")
        cropper = ImageCropper(reference_raster)

    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc=f"Cropping {feature_name}"):
        output_path = Path(output_dir) / f"Crop_{feature_name}_{idx}.tif"

        if is_ndvi:
            cropper.crop_ndvi(
                str(output_path), row["date"], row["lat"], row["lon"], crop_size
            )
        else:
            cropper.crop_image(input_raster, str(output_path), row["lat"], row["lon"], crop_size)

    logger.info(f"{feature_name} cropping completed")


def crop_images_for_testing(
    feature_name: str,
    output_dir: str,
    reference_raster: str,
    input_raster: str,
    crop_size: int = 1,
) -> None:
    """
    Crop images for all grid points (for test/inference)

    Args:
        feature_name: Name of the feature
        output_dir: Directory to save cropped images
        reference_raster: Path to reference raster
        input_raster: Path to input raster
        crop_size: Crop window half-size
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calculate grid coordinates
    width_step = (DataConstants.BOUNDARY_EAST - DataConstants.BOUNDARY_WEST) / (
        DataConstants.GRID_WIDTH - 1
    )
    height_step = (DataConstants.BOUNDARY_NORTH - DataConstants.BOUNDARY_SOUTH) / (
        DataConstants.GRID_HEIGHT - 1
    )

    width_coords = [
        round(DataConstants.BOUNDARY_WEST + width_step * i, 7)
        for i in range(DataConstants.GRID_WIDTH)
    ]
    height_coords = [
        round(DataConstants.BOUNDARY_NORTH - height_step * i, 7)
        for i in range(DataConstants.GRID_HEIGHT)
    ]

    # Check if already processed
    expected_count = DataConstants.GRID_HEIGHT * DataConstants.GRID_WIDTH
    existing_files = list(Path(output_dir).glob("*.tif"))

    if len(existing_files) == expected_count:
        logger.info(f"{feature_name} test crop images already exist")
        return

    logger.info(f"Creating {feature_name} test crop images")

    cropper = ImageCropper(reference_raster)
    coordinate_combinations = product(height_coords, width_coords)

    for num, (lat, lon) in tqdm(
        enumerate(coordinate_combinations),
        total=expected_count,
        desc=f"Cropping {feature_name} (test)",
    ):
        output_path = Path(output_dir) / f"Crop_{feature_name}_{num}.tif"
        cropper.crop_image(input_raster, str(output_path), lat, lon, crop_size)

    logger.info(f"{feature_name} test cropping completed")


def convert_tif_to_npy(
    feature_name: str,
    data_type: str,
    crop_dir: str,
    output_dir: str,
    image_size: int = 64,
) -> np.ndarray:
    """
    Convert cropped .tif images to .npy array

    Args:
        feature_name: Name of the feature
        data_type: 'train' or 'test'
        crop_dir: Directory with cropped .tif files
        output_dir: Directory to save .npy file
        image_size: Target image size (will be resized)

    Returns:
        NumPy array of images
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / f"{feature_name}_{data_type}.npy"

    if output_path.exists():
        logger.info(f"{feature_name}_{data_type}.npy already exists")
        return np.load(output_path)

    logger.info(f"Creating {feature_name}_{data_type}.npy")

    # Get all .tif files
    tif_files = natsort.natsorted(list(Path(crop_dir).glob("*.tif")))

    if len(tif_files) == 0:
        raise FileNotFoundError(f"No .tif files found in {crop_dir}")

    # Process images
    images = []
    for tif_file in tqdm(tif_files, desc=f"Converting {feature_name}"):
        # Read image
        img = cv2.imread(str(tif_file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (image_size, image_size))

        # Normalize to 0-1
        img = img / 255.0

        images.append(img)

    # Convert to array
    data = np.array(images, dtype=np.float32)

    # Save
    np.save(output_path, data)
    logger.info(f"{feature_name}_{data_type}.npy saved: shape {data.shape}")

    return data
