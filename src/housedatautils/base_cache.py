import json
import logging
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional, Any
from .logging_config import setup_logging


class BaseCache:
    DEFAULT_METADATA = {
        "housing": {},
        "geo": {}
    }

    def __init__(self, loader_type: str,
                 metadata_path: Path,
                 format: str,
                 save_method: Callable[[pd.DataFrame, str], None],
                 load_method: Callable[[str], pd.DataFrame]) -> None:
        """
        Initialize the BaseCache.

        Parameters:
        - loader_type (str): The type of data loader ('housing', 'geo', etc.).
        - metadata_path (Path): The path to the metadata JSON file.
        - format (str): The file format for caching ('pkl', 'parquet').
        - save_method (Callable): Function to save data to file.
        - load_method (Callable): Function to load data from file.
        """
        setup_logging()
        self._logger = logging.getLogger(__name__)
        self._loader_type = (loader_type.value
                             if isinstance(loader_type, Enum)
                             else loader_type)
        self._metadata_path = metadata_path
        self._format = format
        self._save_method = save_method
        self._load_method = load_method
        self._metadata: Dict[str, Any] = self._load_metadata()

    def cache_data(self,
                   data: pd.DataFrame,
                   path: Optional[Path] = None) -> None:
        """
        Cache the given data to a file.

        Parameters:
        - data (pd.DataFrame): Data to cache.
        - path (Optional[Path]): Optional path to cache the data. If not
        provided, a default path is used.
        """
        if path is None:
            path = f"preload/{self._loader_type}_data.{self._format}"

        self._save_method(data, path)
        self._update_and_save_metadata(self._format, str(path))

    async def load_data(self,
                        path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """
        Load data from a cached file.

        Parameters:
        - path (Optional[Path]): Optional path to load the data from. If not
        provided, the default path from metadata is used.

        Returns:
        - pd.DataFrame or None: Loaded data, or None if file does not exist.
        """
        if path is None:
            path = self._metadata.get(self._loader_type, {}).get(self._format)
        self._logger.info(f"Cache path obtained from: {path}")
        if path:
            return self._load_method(path)
        return None

    def _update_and_save_metadata(self, format: str, cache_path: str) -> None:
        """
        Update and save metadata to a JSON file. This method first updates the
        internal metadata dictionary to include the new caching information
        and then writes this updated dictionary to a JSON file.

        Parameters:
        - format (str): The format in which the data was cached ('pkl',
        'parquet').
        - cache_path (str): The full path where the data was cached.
        """
        self._update_metadata(format, cache_path)
        self._save_metadata()

    def _update_metadata(self, format: str, cache_path: str) -> None:
        """
        Update the internal metadata dictionary with new caching information.

        This updates the metadata to include the full path to where the data
        is cached, and it also updates a timestamp to indicate when this
        caching was done. The metadata is organized by the type of data loader
        ('housing', 'geo', etc.) and the format ('pkl', 'parquet').

        Parameters:
        - format (str): The format in which the data was cached ('pkl',
        'parquet').
        - cache_path (str): The full path where the data was cached.
        """
        timestamp = str(pd.Timestamp.now())
        if self._loader_type not in self._metadata:
            self._metadata[self._loader_type] = {}
        self._metadata[self._loader_type][format] = cache_path
        self._metadata[self._loader_type]['timestamp'] = timestamp

    def _save_metadata(self) -> None:
        """
        Save the metadata information which includes paths to the cached data
        files (both pickle and parquet), and the timestamp indicating when
        the caching was done.

        Parameters:
        - cache_path (str): The path where the data was cached.
        - format (str): The format in which the data was cached ('pkl' or
          'parquet').
        """
        with open(self._metadata_path, 'w') as file:
            json.dump(self._metadata, file)

    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load the metadata information from the metadata file. This metadata
        includes paths to the cached data files and the timestamp indicating
        when the caching was done.

        Returns:
        - dict: Dictionary containing metadata information. If metadata file
          doesn't exist, it returns an empty dictionary.
        """
        try:
            with open(self._metadata_path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return self.DEFAULT_METADATA
