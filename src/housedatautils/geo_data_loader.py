"""
GeoDataJSONLoader Module
========================

This module provides the GeoDataJSONLoader class which is designed to load
GeoJSON data from a specific directory. The loader provides functionalities
to fetch data based on various conditions like postcode zones, specific
postcodes, and London regions.

Classes:
    - GeoDataJSONLoader: For efficiently loading and preprocessing the GeoJSON
      data.

Usage Example:
-------------
from this_module_name import GeoDataJSONLoader

loader = GeoDataJSONLoader(path='path_to_data')
data = loader._load_data_by_prefix('WA')
"""

import pandas as pd

from .base_data_loader import BaseDataLoader, LoaderType
from typing import Optional, Callable
from pandarallel import pandarallel


class GeoDataJSONLoader(BaseDataLoader):
    """
    GeoDataJSONLoader is a utility class for loading GeoJSON data.

    Inherits from BaseDataLoader and extends its functionalities for more
    specific GeoJSON data operations.

    Attributes:
    -----------
    - loader_type (LoaderType): The type of loader (default is GEO).
    - read_method (Callable): Method to read data if any provided.

    Methods:
    --------
    - _process_data(): Loads and preprocesses the data asynchronously.
    """

    def __init__(self,
                 loader_type: LoaderType = LoaderType.GEO,
                 read_method: Optional[Callable] = None):
        """
        Initializes GeoDataJSONLoader with given loader type and read method.

        Parameters:
        - loader_type (LoaderType): Type of the loader. Defaults to
          LoaderType.GEO.
        - read_method (Optional[Callable]): Optional data reading method.
        """
        super().__init__(loader_type=loader_type,
                         read_method=read_method)

        pandarallel.initialize()

    async def _process_data(self):
        """
        Load and preprocess the data asynchronously.

        Returns:
        --------
        DataFrame: Combined and processed data.
        """

        lst = [self.data]
        if not self._data_cached or self.data is None:
            lst = await self._load_data("geo_data_paths")

        combined_data = pd.concat(lst, ignore_index=True)
        combined_data = combined_data.rename(columns={'postcodes': 'Postcode'})

        # Extract postcode prefixes to read geometry file names
        combined_data['Postcode_prefix'] = combined_data['Postcode'].apply(
            lambda x: x[0] if (isinstance(x, str) and x[1].isdigit() and
                               x[:2] not in self._NON_ENGLAND_POSTCODES)
                           else (x[:2] if isinstance(x, str) else 'Unknown'))

        self.data = combined_data

        return combined_data
