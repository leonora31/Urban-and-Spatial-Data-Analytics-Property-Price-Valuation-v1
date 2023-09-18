import unittest
import asyncio
import pandas as pd
import geopandas as gpd
from unittest.mock import patch, mock_open
from pathlib import Path
from src.housedatautils import BaseDataLoader, LoaderType
from src.housedatautils import BaseCache


class TestBaseDataLoader(unittest.IsolatedAsyncioTestCase):
    """Test cases for the BaseDataLoader class."""

    async def setUp(self):
        """Set up the test case, executed before each test."""
        self.loop = asyncio.get_event_loop()

    def test_init_with_housing_type(self):
        """Test if BaseDataLoader initializes with default housing type."""
        data_loader = BaseDataLoader(loader_type=LoaderType.HOUSING)
        self.assertEqual(data_loader._loader_type, LoaderType.HOUSING)
        self.assertEqual(data_loader._extension, data_loader._CSV_EXTENSION)
        self.assertEqual(data_loader._read_method, pd.read_csv)

    def test_init_with_geo_type(self):
        """Test if BaseDataLoader initializes with geo type."""
        data_loader = BaseDataLoader(loader_type=LoaderType.GEO)
        self.assertEqual(data_loader._loader_type, LoaderType.GEO)
        self.assertEqual(data_loader._extension,
                         data_loader._GEOJSON_EXTENSION)
        self.assertEqual(data_loader._read_method, gpd.read_file)

    def test_init_with_invalid_type(self):
        """Test if BaseDataLoader raises an error with an invalid type."""
        with self.assertRaises(ValueError):
            BaseDataLoader(loader_type='invalid_type')

    def test_load_config(self):
        """Test the internal _load_config method."""
        data_loader = BaseDataLoader()
        config = data_loader._paths
        self.assertIn('data_paths', config)
        self.assertIn('geo_data_paths', config)

    async def test_process_data(self):
        """Test the internal _process_data method."""
        with self.assertRaises(NotImplementedError):
            data_loader = BaseDataLoader()
            await data_loader._process_data()


# Define the path for the metadata file used in tests
METADATA_PATH = Path("test_metadata.json")


class TestBaseCache(unittest.TestCase):
    """Test cases for the BaseCache class."""

    def setUp(self):
        """Set up the test case, executed before each test."""
        self.base_cache = BaseCache(
            loader_type=LoaderType.HOUSING,
            metadata_path=METADATA_PATH,
            format="pkl",
            save_method=pd.DataFrame.to_pickle,
            load_method=pd.read_pickle
        )

    def tearDown(self):
        """Clean up after the test case, executed after each test."""
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

        test_csv_path = Path("housing_data.csv")
        if test_csv_path.exists():
            test_csv_path.unlink()

    def test_initialization(self):
        """Test if BaseCache initializes properly."""
        self.assertEqual(self.base_cache.DEFAULT_METADATA,
                         {"housing": {},
                          "geo": {}})
        self.assertEqual(self.base_cache._loader_type,
                         LoaderType.HOUSING.value)
        self.assertEqual(self.base_cache._load_method, pd.read_pickle)

    def test_cache_data(self):
        """Test the cache data method."""
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        cache_path = "test_cache_data" + self.base_cache._format
        self.base_cache._save_method(data, path=cache_path)
        self.assertTrue(Path(cache_path).exists())

    def test_load_data(self):
        """Test the load data method."""
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        cache_path = "test_cache_data" + self.base_cache._format
        loaded_data = self.base_cache._load_method(cache_path)
        self.assertTrue(data.equals(loaded_data))

    def test_load_metadata(self):
        """Test the internal _load_metadata method."""
        with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
            self.assertEqual(self.base_cache._load_metadata(),
                             {"key": "value"})

        self.assertEqual(self.base_cache._load_metadata(),
                         self.base_cache.DEFAULT_METADATA)


if __name__ == "__main__":
    unittest.main()
