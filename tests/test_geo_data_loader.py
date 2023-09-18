import unittest
import asyncio
import geopandas as gpd
import os

from src.housedatautils import GeoDataJSONLoader, LoaderType


class TestGeoDataJSONLoader(unittest.IsolatedAsyncioTestCase):
    """Test cases for the BaseDataLoader class."""

    async def setUp(self):
        """Set up the test case, executed before each test."""
        self.loop = asyncio.get_event_loop()

    def test_init_with_geo_type(self):
        """Test if BaseDataLoader initializes with geo type."""
        data_loader = GeoDataJSONLoader()
        self.assertEqual(data_loader._loader_type, LoaderType.GEO)
        self.assertEqual(data_loader._extension,
                         data_loader._GEOJSON_EXTENSION)
        self.assertEqual(data_loader._read_method, gpd.read_file)

    def test_init_with_invalid_type(self):
        """Test if BaseDataLoader raises an error with an invalid type."""
        with self.assertRaises(ValueError):
            GeoDataJSONLoader(loader_type='invalid_type')

    def test_load_config(self):
        """Test the internal _load_config method."""
        data_loader = GeoDataJSONLoader()
        config = data_loader._paths
        self.assertIn('data_paths', config)
        self.assertIn('geo_data_paths', config)

    async def test_process_data_geo(self):
        """Test the internal _process_data method of GeoDataJSONLoader."""
        geo_loader = GeoDataJSONLoader()
        print(os.getcwd())
        gdf = gpd.read_file('./tests/sample.geojson')

        geo_loader.data = gdf
        geo_loader._data_cached = True

        # # Call the method
        result = await geo_loader._process_data()
        print(result)

        # # Assertions
        self.assertTrue(isinstance(result, gpd.GeoDataFrame))
        self.assertTrue('Postcode' in result.columns)
        self.assertTrue('mapit_code' in result.columns)
        self.assertTrue('geometry' in result.columns)
        self.assertFalse(result['Postcode'].isin(
            geo_loader._NON_ENGLAND_POSTCODES).any()
        )


if __name__ == "__main__":
    unittest.main()
