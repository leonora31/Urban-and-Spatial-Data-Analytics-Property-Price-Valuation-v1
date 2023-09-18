"""BaseDataLoader

This module provides an asynchronous data loading and caching framework.

The BaseDataLoader class supports loading data from multiple file formats,
caching for faster loads, and filtering data. It is designed to be extended
by subclassing to handle different data types.

Key classes:

- BaseDataLoader: Abstract base class for async data loading operations.

- BaseCache: Handles caching data to files and reading cached data.

- LoaderType (Enum): Supported data loader types like HOUSING, GEO.

Usage:

- Subclass BaseDataLoader for a particular data type.

- Override `_process_data` to implement data preprocessing.

- Call `load_*` methods to load and filter data for specific cases.

- Data is lazy loaded and cached for faster subsequent access.

The module provides a general framework for:

- Async data loading from multiple files and sources.
- Caching data to speed up subsequent loads.
- Filtering data based on geographic regions or other properties.
- Extending for new data types by subclassing BaseDataLoader.

Dependencies:
- asyncio: For asynchronous I/O operations.
- pandas: For tabular data processing.
- geopandas: For geographic data.
- pickle: For serializing and deserializing Python objects.
- json: For reading and writing JSON files.
- logging: For logging messages to console and file.
- pathlib: For working with file paths.
- enum: For defining enumerations.
- typing: For type hints.
- time: For measuring elapsed time.
- os: For working with operating system.

Example:
    >>> from housedatautils.base_data_loader_refactor import BaseDataLoader
    >>> loader = BaseDataLoader()
    >>> await loader._load_or_process_data()
    >>> loader.data
    >>> loader.load_inner_london_data()
    >>> loader.load_outer_london_data()

"""

import asyncio
import json
import logging
import geopandas as gpd
import pandas as pd
import pickle
import time
import os

from pathlib import Path
from enum import Enum
from typing import Callable, Dict, List, Optional
from .base_cache import BaseCache
from .logging_config import setup_logging


class LoaderType(Enum):
    """Enumeration for different types of data loaders."""
    HOUSING = 'housing'
    GEO = 'geo'


class LondonZone(Enum):
    """Enumeration for different types of London data."""
    INNER = "Inner"
    OUTER = "Outer"


class BaseDataLoader:

    _INNER_LONDON_PREFIXES = ['E', 'EC', 'N', 'NW', 'SE', 'SW', 'W', 'WC']
    _OUTER_LONDON_PREFIXES = ['BR', 'CR', 'DA', 'EN', 'HA', 'IG', 'KT', 'RM',
                              'SM', 'TW', 'UB', 'WD']

    # Create lists of outward postcodes for Scotland, Wales, and NI
    _SCOTLAND_POSTCODES = ["AB", "DD", "DG", "EH", "FK", "G", "HS", "IV",
                           "KA", "KW", "KY", "ML", "PA", "PH", "TD", "ZE"]
    _WALES_POSTCODES = ["CF", "LD", "LL", "NP", "SA", "SY"]
    _NORTHERN_IRELAND_POSTCODES = ["BT"]

    # Combine all the lists into one
    _NON_ENGLAND_POSTCODES = _SCOTLAND_POSTCODES + _WALES_POSTCODES + \
                             _NORTHERN_IRELAND_POSTCODES

    _CSV_EXTENSION = '.csv'
    _GEOJSON_EXTENSION = '.geojson'

    # Define constants for data formats
    _PICKLE_FORMAT = "pkl"
    _PARQUET_FORMAT = "parquet"
    _DEFAULT_CONDITION = lambda x: True

    def __init__(self,
                 loader_type: LoaderType = LoaderType.HOUSING,
                 read_method: Optional[Callable] = None
                 ):
        """
        Initializes the BaseDataLoader with configuration path, caching
        preference, and loader type.


        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file, by default "config.json"
        cached_data : bool, optional
            Whether to load cached data, by default False
        loader_type : LoaderType, optional
            Type of data loader, by default LoaderType.HOUSING
        read_method : Callable, optional
            Method to read the data, by default None
        """

        self.data = None
        setup_logging()
        self._logger = logging.getLogger(__name__)
        self._logger.info("BaseDataLoader initialized")

        # Check if the current working directory contains 'src'
        if 'src' in os.getcwd().split("/"):
            default_path = Path('jsons/config.json')
        else:
            # Relative to this script's location
            default_path = Path(os.getcwd()+'/src/jsons/config.json')

        self._paths = self._load_config(default_path)
        self._metadata_path = Path.cwd() / "jsons/metadata.json"
        self._data_cached = False
        self._data_preprocessed = False

        self._loader_type = loader_type
        self._read_method = read_method

        if self._loader_type == LoaderType.HOUSING:
            if not self._read_method:
                self._read_method = pd.read_csv
                self._save_method = pd.DataFrame.to_parquet
                self._extension = self._CSV_EXTENSION

        elif self._loader_type == LoaderType.GEO:
            if not self._read_method:
                self._read_method = gpd.read_file
                # self._save_method = pd.DataFrame.to_parquet
                self._save_method = gpd.GeoDataFrame.to_parquet
                self._extension = self._GEOJSON_EXTENSION
        else:
            raise ValueError(f"Invalid loader type: {self._loader_type}")

    def _load_config(self,
                     config_path: Path = Path('jsons/config.json')) -> Dict:
        """
        Load configuration data for file paths from a JSON file.


        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file, by default "config.json"

        Returns
        -------
        Dict
            Dictionary containing configuration data.
        """
        with open(config_path, "r") as f:
            return json.load(f)

    def __await__(self):
        """
        Special method that allows the BaseDataLoader class to be used with
        the `await` keyword. It ensures that data loading or processing is
        done asynchronously during object initialization.

        Returns:
        - Generator: An awaitable generator, which is the result of
        `_load_or_process_data` method.
        """
        return self._load_or_process_data().__await__()

    @staticmethod
    def ensure_data_loaded(func):
        """
        Decorator to ensure data is loaded or processed before executing a
        method.

        Parameters:
        - func (callable): The asynchronous function to wrap and check for
        loaded data before execution.

        Returns:
        - callable: The wrapped function.
        """
        async def wrapper(instance, *args, **kwargs):
            await instance._load_or_process_data()
            return await func(instance, *args, **kwargs)
        return wrapper

    async def _load_or_process_data(self, load_cached_data=True):
        """
        Load or process data based on the provided parameters and available
        cache.

        Parameters:
        - load_cached_data (bool, optional): Flag to indicate whether to
        attempt loading from cached data. Defaults to True.

        Returns:
        - BaseDataLoader: Returns the instance of the class with the loaded or
        processed data populated.
        """
        if (load_cached_data and not self._data_cached and
            not self._data_preprocessed):
            await self._attempt_data_loading_sequence()

        await asyncio.sleep(1)
        return self

    async def _cache_data(self):
        self._pickle_cache.cache_data(self.data)
        self._logger.info("Pickle data cached")
        
        self._parquet_cache.cache_data(self.data)
        self._logger.info("Parquet data cached")

    async def _attempt_data_loading_sequence(self):
        """
        Attempt to sequentially load data from different supported formats
        until successful.

        This method tries to load data from each supported format one by one
        until it successfully loads the data or exhausts all options. If all
        options are exhausted, it will preprocess the data from the source
        files.
        """
        self._pickle_cache = BaseCache(self._loader_type,
                                       self._metadata_path,
                                       self._PICKLE_FORMAT,
                                       pd.DataFrame.to_pickle,
                                       pd.read_pickle)

        self._parquet_cache = BaseCache(self._loader_type,
                                        self._metadata_path,
                                        self._PARQUET_FORMAT,
                                        self._save_method,
                                        pd.read_parquet)

        loaded = await self._try_loading_data(self._PICKLE_FORMAT,
                                              self._pickle_cache.load_data)

        if loaded:
            await self._cache_data()
            return

        loaded = await self._try_loading_data(self._PARQUET_FORMAT,
                                              self._parquet_cache.load_data)
        if loaded:
            await self._cache_data()
            return

        await self._preprocess_data()
        self._data_preprocessed = True

    async def _try_loading_data(self, format_name, load_method):
        """
        Attempt to load data using the specified format and load method.

        Parameters:
        - format_name (str): The name of the format to try for loading data.
        - load_method (Callable): The asynchronous method to call for loading
        the data.

        Returns:
        - bool: True if data was successfully loaded, False otherwise.
        """
        try:
            self.data = await load_method()
            if self.data is not None:
                self._logger.info(f"Loaded cached data using {format_name}.")
                self._data_cached = True
                return True

            raise FileNotFoundError("Data not loaded correctly from" +
                                    f" {format_name}.")

        except (FileNotFoundError, EOFError, pickle.UnpicklingError,
                OSError, IsADirectoryError) as e:
            self._logger.error(f"Failed to load data using {format_name}. " +
                               f"Error: {e}")
            return False

    async def _read_file(self,
                         file_path: str,
                         condition: Callable[[str], bool],
                         total_files: int = 0,
                         current_file: int = 0) -> Optional[pd.DataFrame]:
        """
        Asynchronously reads a file based on the given conditions.

        Parameters:
        - file_path (str): Path to the file to be read.
        - condition (Callable[[str], bool]): A function to determine if the
        file should be loaded.
        - total_files (int, optional): Total number of files to be loaded.
        Used for printing purposes. Defaults to 0.
        - current_file (int, optional): Index of the current file being loaded
        Used for printing purposes. Defaults to 0.

        Returns:
        - Optional[pd.DataFrame]: Data from the file if successful, otherwise
        None.

        Notes:
        - The file extension to be used for reading is determined by the
        instance attribute `self._extension`.
        - The method to read the file is determined by the instance attribute
        `self._read_method`.
        """
        try:
            name = file_path.split("/")[-1].split(".")[0]

            # Check if name starts with any of the non-england postcodes
            if any(
                name.startswith(
                    prefix
                ) for prefix in self._NON_ENGLAND_POSTCODES
            ):
                return None
            if file_path.endswith(self._extension) and condition(file_path):
                df = self._read_method(file_path)
                self._logger.info(f"File #{current_file}/{total_files} {name}"
                                  "was loaded")
                return df
        except Exception as e:
            self._logger.error(f"Error reading {file_path}: {e}")
        return None

    async def _validate_data_type(self, data_type: str) -> None:
        """
        Validate if the given data_type exists in self._paths.

        Parameters:
        - data_type (str): The type of data to load.

        Raises:
        - ValueError: If the specified data_type is not found in the instance
        attribute `self._paths`.
        """
        if data_type not in self._paths:
            raise ValueError(f"Invalid data type: {data_type}. Expected one" +
                             f" of {list(self._paths.keys())}.")

    def _compute_file_list(self,
                           data_type: str,
                           condition: Callable[[str], bool]) -> List[str]:
        """
        Compute a list of file paths based on the data_type and condition.

        Parameters:
        - data_type (str): The type of data to load.
        - condition (Callable[[str], bool]): A function to filter the files.

        Returns:
        - List[str]: A list of file paths.
        """
        specific_paths = self._paths[data_type]
        if "geo_data_json" in specific_paths:
            path = specific_paths["geo_data_json"]
            return [f"{path}{filename}" for filename in os.listdir(path)]

        return [file for file in specific_paths.values()
                if file.endswith(self._extension) and condition(file)]

    async def _load_files(self,
                          file_list: List[str],
                          condition: Callable[[str], bool]
                          ) -> List[Optional[object]]:
        """
        Asynchronously load files based on the given list and condition.

        Parameters:
        - file_list (List[str]): A list of file paths.
        - condition (Callable[[str], bool]): A function to filter the files.

        Returns:
        - List[Optional[object]]: A list of loaded files or None if errors
        occurred.
        """
        total_files = len(file_list)
        results = await asyncio.gather(*[self._read_file(filename,
                                                         condition,
                                                         total_files,
                                                         idx+1)
                                       for idx, filename in enumerate(
                                           file_list)],
                                       return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self._logger.error(f"An error occurred: {result}")

        return [
            result for result in results if not isinstance(result, Exception)
        ], total_files

    async def _load_data(self, data_type: str,
                         condition: Callable[[str], bool] = _DEFAULT_CONDITION
                         ) -> List[Optional[object]]:
        """
        Asynchronously loads data files based on the specified conditions and
        file extensions.

        Parameters:
        - data_type (str): The type of data to load.
        - condition (Callable[[str], bool], optional): A function that returns
        a boolean value. Only files for which this function returns True will
        be loaded. By default, all files are loaded.

        Returns:
        - List[Optional[object]]: A list of DataFrames corresponding to the
        loaded data files.
        """
        start_time = time.time()

        await self._validate_data_type(data_type)

        file_list = self._compute_file_list(data_type, condition)

        dataframes, total_files = await self._load_files(file_list, condition)

        end_time = time.time()
        elapsed_time = end_time - start_time
        self._logger.info(f"Loaded {total_files} files in {elapsed_time:.2f}" +
                          " seconds.")

        return dataframes

    async def _process_data(self):
        """
        Placeholder method for data preprocessing steps. To be overridden in
        derived classes.
        """
        raise NotImplementedError("This method should be implemented in " +
                                  "inheriting classes.")

    async def _preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess the data asynchronously. This method calls
        `_process_data` for the data processing steps and then caches
        the data.
        """
        pickle_data = self._pickle_cache.load_data()
        parquet_data = self._parquet_cache.load_data()
        self.data = pickle_data or parquet_data

        # Call the method for data preprocessing
        self.data = await self._process_data()
        await self._cache_data()

        return self.data

    @ensure_data_loaded
    async def _filter_data(self, column_name: str,
                           values: List[str]) -> pd.DataFrame:
        """
        Generic filtering logic.
        """
        return self.data[self.data[column_name].isin(values)]

    @ensure_data_loaded
    async def load_inner_london_data(self) -> pd.DataFrame:
        """
        Load and return data specific to Inner London.
        """
        return await self._filter_data('Postcode_prefix',
                                       self._INNER_LONDON_PREFIXES)

    @ensure_data_loaded
    async def load_outer_london_data(self) -> pd.DataFrame:
        """
        Load and return data specific to Outer London.
        """
        return await self._filter_data('Postcode_prefix',
                                       self._OUTER_LONDON_PREFIXES)

    @ensure_data_loaded
    async def load_data_london(self) -> pd.DataFrame:
        """
        Load and return data specific to both Inner and Outer London.
        """
        inner_data = await self.load_inner_london_data()
        outer_data = await self.load_outer_london_data()

        # Combine the inner and outer data
        return pd.concat([inner_data, outer_data], ignore_index=True)

    @ensure_data_loaded
    async def load_major_postcode_zone(self,
                                       major_zones: List[str]) -> pd.DataFrame:
        """
        Load and return data specific to the postcode zone.
        """
        temp_data = self.data.copy()
        temp_data['Major_Zone'] = temp_data['Postcode'].str.split(' ').str[0]
        return await self._filter_data('Major_Zone', major_zones)

    @ensure_data_loaded
    async def load_uk_data(self) -> pd.DataFrame:
        """
        Load and return all the data without any specific filter.

        Returns:
        - pd.DataFrame: The entire dataset without any specific filtering.
        """
        return self.data
