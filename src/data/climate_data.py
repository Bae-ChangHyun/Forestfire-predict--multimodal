"""Climate data collection and processing module"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
import logging

from ..utils.constants import DataConstants, APIConstants

logger = logging.getLogger(__name__)


class ClimateDataCollector:
    """Collect climate data from ASOS API"""

    def __init__(self, api_key: str):
        """
        Initialize climate data collector

        Args:
            api_key: ASOS API key
        """
        self.api_key = api_key
        self.base_url = APIConstants.ASOS_BASE_URL

    def crawl_asos_hourly(
        self, date: str, location_code: int
    ) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Crawl hourly ASOS weather data for a specific location and time

        Args:
            date: Date in format 'YYYYMMDDHH'
            location_code: ASOS location code

        Returns:
            Tuple of (time, humidity, windspeed, rainfall, temperature)
            Returns (None, None, None, None, None) if data not found
        """
        # Parse date
        start_dt = datetime.strptime(date[:8], "%Y%m%d")
        start_hh = datetime.strptime(date[8:], "%H")
        end_hh = start_hh + timedelta(hours=1)

        # Handle day change
        if end_hh.strftime("%H:%M:%S").split(":")[0] == "00":
            end_dt = start_dt + timedelta(days=1)
        else:
            end_dt = start_dt

        # Format parameters
        params = {
            "serviceKey": self.api_key,
            "pageNo": "1",
            "numOfRows": "10",
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "HR",
            "startDt": start_dt.strftime("%Y%m%d"),
            "startHh": start_hh.strftime("%H"),
            "endDt": end_dt.strftime("%Y%m%d"),
            "endHh": end_hh.strftime("%H"),
            "stnIds": str(location_code),
        }

        # Retry logic
        for retry in range(APIConstants.MAX_RETRIES):
            try:
                response = requests.get(
                    self.base_url,
                    params=params,
                    verify=False,
                    timeout=APIConstants.REQUEST_TIMEOUT,
                )
                response.raise_for_status()

                json_obj = json.loads(response.content)
                item = json_obj.get("response", {}).get("body", {}).get("items", {}).get("item", [{}])[0]

                if item:
                    return (
                        item.get("tm"),
                        item.get("hm"),  # humidity
                        item.get("ws"),  # windspeed
                        item.get("rn"),  # rainfall
                        item.get("ta"),  # temperature
                    )
                else:
                    logger.warning(f"No data found for {date}, location {location_code}")
                    time.sleep(APIConstants.RETRY_DELAY)

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (retry {retry + 1}/{APIConstants.MAX_RETRIES}): {e}")
                time.sleep(APIConstants.RETRY_DELAY)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"JSON parsing error: {e}")
                time.sleep(APIConstants.RETRY_DELAY)

        logger.error(f"Failed to get data for {date}, location {location_code} after all retries")
        return None, None, None, None, None


class FutureWeatherCollector:
    """Collect future weather forecast data"""

    def __init__(self, api_key: str):
        """
        Initialize future weather collector

        Args:
            api_key: Weather API key
        """
        self.api_key = api_key
        self.base_url = APIConstants.VILAGE_FCST_BASE_URL

    def crawl_ultra_short_forecast(
        self, date: str, time: str, grid_x: int, grid_y: int
    ) -> Optional[List[Dict]]:
        """
        Crawl ultra short-term forecast data

        Args:
            date: Date in format 'YYYYMMDD'
            time: Time in format 'HHMM'
            grid_x: Grid X coordinate
            grid_y: Grid Y coordinate

        Returns:
            List of forecast items or None if failed
        """
        params = {
            "serviceKey": self.api_key,
            "pageNo": "1",
            "numOfRows": "1000",
            "dataType": "JSON",
            "base_date": date,
            "base_time": time,
            "nx": str(grid_x),
            "ny": str(grid_y),
        }

        for retry in range(APIConstants.MAX_RETRIES):
            try:
                response = requests.get(
                    self.base_url,
                    params=params,
                    verify=False,
                    timeout=APIConstants.REQUEST_TIMEOUT,
                )
                response.raise_for_status()

                json_obj = json.loads(response.content)
                items = json_obj["response"]["body"]["items"]["item"]
                return items

            except requests.exceptions.RequestException as e:
                logger.error(f"Network error (retry {retry + 1}): {e}")
                time.sleep(APIConstants.RETRY_DELAY)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"JSON parsing error: {e}")
                time.sleep(APIConstants.RETRY_DELAY)

        logger.error(f"Failed to get forecast data after all retries")
        return None


def load_asos_from_csv(
    asos_csv_path: str, date_str: str, location_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Load ASOS data from pre-downloaded CSV file

    Args:
        asos_csv_path: Path to ASOS CSV file directory
        date_str: Date string in format 'YYYYMMDDHH'
        location_df: DataFrame with location information

    Returns:
        DataFrame with climate data
    """
    year = date_str[:4]
    csv_file = Path(asos_csv_path) / f"ASOS_Hr_{year}.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"ASOS CSV file not found: {csv_file}")

    # Load CSV
    df_asos_hr = pd.read_csv(csv_file, low_memory=False)

    # Parse date
    date_obj = datetime.strptime(date_str, "%Y%m%d%H")
    date_formatted = date_obj.strftime("%Y-%m-%d %H:%M")

    # Filter by date
    df_filtered = df_asos_hr[df_asos_hr["일시"] == date_formatted].reset_index(drop=True)

    # Add longitude and latitude
    df_filtered["longitude"] = df_filtered.apply(
        lambda row: (
            location_df.loc[location_df["지점번호"] == row["지점"], "lon"].values[0]
            if location_df.loc[location_df["지점번호"] == row["지점"], "lon"].shape[0] != 0
            else np.nan
        ),
        axis=1,
    )
    df_filtered["latitude"] = df_filtered.apply(
        lambda row: (
            location_df.loc[location_df["지점번호"] == row["지점"], "lat"].values[0]
            if location_df.loc[location_df["지점번호"] == row["지점"], "lat"].shape[0] != 0
            else np.nan
        ),
        axis=1,
    )

    # Rename columns
    df_filtered = df_filtered[
        ["지점", "longitude", "latitude", "일시", "습도(%)", "풍속(m/s)", "강수량(mm)", "기온(°C)"]
    ]
    df_filtered.columns = ["loc_info", "lon", "lat", "time", "humidity", "windspeed", "rainfall", "temp"]

    return df_filtered
