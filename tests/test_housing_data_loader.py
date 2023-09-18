import unittest
import asyncio
import pandas as pd

from unittest.mock import patch, AsyncMock
from src.housedatautils import HousingDataLoader, LoaderType


class TestHousingDataLoader(unittest.IsolatedAsyncioTestCase):
    """Test cases for the HousingDataLoader class."""

    async def setUp(self):
        """Set up the test case, executed before each test."""
        self.loop = asyncio.get_event_loop()

    def test_init(self):
        """Test if HousingDataLoader initializes with default housing type."""
        data_loader = HousingDataLoader()
        self.assertEqual(data_loader._loader_type, LoaderType.HOUSING)
        self.assertEqual(data_loader._extension, data_loader._CSV_EXTENSION)
        self.assertEqual(data_loader._read_method, pd.read_csv)

    def test_init_with_invalid_type(self):
        """Test if HousingDataLoader raises an error with an invalid type."""
        with self.assertRaises(ValueError):
            HousingDataLoader(loader_type='invalid_type')

    def test_load_config(self):
        """Test the internal _load_config method."""
        data_loader = HousingDataLoader()
        config = data_loader._paths
        self.assertIn('data_paths', config)

    async def test_process_data(self):
        """Test the internal _process_data method."""
        data_loader = HousingDataLoader()

        # Mock Data
        mock_merged_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        mock_preprocessed_data = pd.DataFrame(
            {'A': [10, 20, 30], 'B': [40, 50, 60]}
        )

        # Mock the methods called inside _process_data
        with patch.object(data_loader,
                          '_load_and_merge_data',
                          new_callable=AsyncMock
                          ) as mock_merge, patch.object(
                            data_loader,
                            '_perform_data_preprocessing',
                            new_callable=AsyncMock) as mock_preprocess:

            # Set mock return values
            mock_merge.return_value = mock_merged_data
            mock_preprocess.return_value = mock_preprocessed_data

            # Call the method
            result = await data_loader._process_data()

            # Assertions
            self.assertTrue(isinstance(result, pd.DataFrame))
            self.assertEqual(result.iloc[0, 0], 10)

            # Ensure the mocked methods were called
            mock_merge.assert_called_once()
            mock_preprocess.assert_called_once()

    async def test_rename_columns(self):
        """Test renaming columns."""
        data = pd.DataFrame({'Postcode_x': [1, 2],
                             'Average_number_of_parks_or_public_gardens' +
                             '_within_1_000_m_radius': [3, 4]})
        expected = pd.DataFrame({'Postcode': [1, 2],
                                'Avg num of parks': [3, 4]})

        loader = HousingDataLoader()
        result = await loader._rename_columns(data)

        pd.testing.assert_frame_equal(result, expected)

    def test_merge_with_airports(self):
        """Test mergin with airports."""

        # Mock data
        data = pd.DataFrame({
            'Postcode': ['SW3 3DS', 'NW1 6XX', 'W9 3BQ'],
            'Property_Value': [350000, 830000, 420000]
        })

        properties_airports = pd.DataFrame({
            'Postcode': ['SW3 3DS', 'NW1 6XX', 'W9 3BQ'],
            'Airport_Distance': [13.7, 10.3, 15.1]
        })

        expected = pd.DataFrame({
            'Postcode': ['SW3 3DS', 'NW1 6XX', 'W9 3BQ'],
            'Property_Value': [350000, 830000, 420000],
            'Airport_Distance': [13.7, 10.3, 15.1]
        })

        loader = HousingDataLoader()
        result = asyncio.run(loader._merge_with_airports(data,
                                                         properties_airports))

        pd.testing.assert_frame_equal(result, expected)

    def test_clean_data(self):
        """Test cleaning data."""

        # Mock data with more realistic property values
        data = pd.DataFrame({
            'Postcode': ['SW1A 1AA', 'W1A 0AX', 'M1 1AE', 'EC1A 1BB'],
            'Property_Value': [1250000, 950000, 220000, 1100000],
            'Nearest_Station_Distance': [0.5, 2.5, 1.5, 0.7],
            'Average_distance_to_nearest_park_or_public_garden__m_':
            [1200, 800, 3500, 1500],
            'Nearest_Airport_Distance': [15, 22, 18, 19],
            'Postcode_no_space': ['SW1A1AA', 'W1A0AX', 'M11AE', 'EC1A1BB'],
            'Postcode_y': ['SW1A 1AA', 'W1A 0AX', 'M1 1AE', 'EC1A 1BB']
        })

        # Expected data after cleaning
        expected = pd.DataFrame({
            'Postcode': ['SW1A 1AA', 'W1A 0AX', 'M1 1AE', 'EC1A 1BB'],
            'Property_Value': [1250000, 950000, 220000, 1100000],
            'Nearest Station <3 km': [1, 1, 1, 1],
            'Nearest Park <3 km': [1, 1, 0, 1],
            'Nearest Airport <20 km': [1, 0, 1, 1]
        })

        loader = HousingDataLoader()
        result = asyncio.run(loader._clean_data(data))

        pd.testing.assert_frame_equal(result, expected)

    def test_filter_years(self):
        """Test the internal _filter_years method."""
        # Mock data spanning multiple years with real UK postcodes
        data = pd.DataFrame({
            'Postcode': [
                # First occurrence of these postcodes
                'SW1A 1AA', 'W1A 0AX', 'M1 1AE', 'B33 8TH', 'EC1A 1BB',
                'SW1A 1AA', 'W1A 0AX', 'M1 1AE',  # Second occurrence in 2019
                # Appear in 2019 and years outside 2008-2019 range
                'NW1 6XE', 'SE15 5XZ'
            ],
            'Year':
            [2008, 2010, 2009, 2007, 2005, 2019, 2019, 2019, 2004, 2019]
        })

        # Expected data after filtering
        expected = pd.DataFrame({
            'Postcode': ['SW1A 1AA', 'W1A 0AX', 'M1 1AE', 'SW1A 1AA',
                         'W1A 0AX', 'M1 1AE', 'SE15 5XZ'],
            'Year': [2008, 2010, 2009, 2019, 2019, 2019, 2019]
        })

        loader = HousingDataLoader()
        result = asyncio.run(loader._filter_years(data))
        # Resetting the index of the result dataframe
        result = result.reset_index(drop=True)
        pd.testing.assert_frame_equal(result, expected)

    def test_update_crime_data(self):
        """Test the internal _update_crime_data method."""
        # Mock data with real UK postcodes
        data = pd.DataFrame({
            'Postcode': [
                'SW1A 1AA', 'SW1A 1AB', 'SW1A 2AC', 'M1 1AE', 'B33 8TH',
                'SW1A 2AD', 'SW1A 1AF', 'SW1A 1AG', 'SW1A 1AH', 'SW1A 1AJ',
                'OX1 1AB', 'OX1 1AC', 'OX1 1AD'
            ],
            'Year': [2018, 2018, 2018, 2019, 2019, 2019, 2018, 2018, 2018,
                     2018, 2018, 2018, 2018],
            'Number_of_crimes': [5, 7, 6, 10, 15, 8, 9, 3, 4, 2, 6, 7, 8]
        })

        # Expected data after updating with crime buffer data
        expected = pd.DataFrame({
            'Postcode': [
                'SW1A 1AA', 'SW1A 1AB', 'SW1A 2AC', 'M1 1AE', 'B33 8TH',
                'SW1A 2AD', 'SW1A 1AF', 'SW1A 1AG', 'SW1A 1AH', 'SW1A 1AJ',
                'OX1 1AB', 'OX1 1AC', 'OX1 1AD'
            ],
            'Year': [2018, 2018, 2018, 2019, 2019, 2019, 2018, 2018, 2018,
                     2018, 2018, 2018, 2018],
            'Number_of_crimes': [5, 7, 6, 10, 15, 8, 9, 3, 4, 2, 6, 7, 8],
            'Crimes_Buffer': [30, 30, 6, 10, 15, 8, 30, 30, 30, 30, 21, 21, 21]
        })

        loader = HousingDataLoader()
        result = asyncio.run(loader._update_crime_data(data))

        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_calculate_price_features(self):
        """Test the internal _update_crime_data method."""
        loader = HousingDataLoader()

        data = pd.DataFrame({
            'Postcode': ['SE25 4DZ', 'SE25 4DZ', 'W9 3BQ', 'SW18 1JE',
                         'SW18 1JE', 'N12 0NL', 'N12 0NL', 'N12 0NL',
                         'N12 0NL', 'N12 0NL'],
            'Price': [450000.0, 249950.0, 420000.0, 1803571.0, 1175000.0,
                      218750.0, 518333.0, 337500.0, 233750.0, 10000.0],
            'Year': [2013, 2012, 2013, 2017, 2019, 2014, 2017, 2019, 2013,
                     2018],
            'Nearest_Station': ['Elephant & Castle', 'Elephant & Castle',
                                'Paddington', 'King\'s Cross', 'King\'s Cross',
                                'King\'s Cross', 'King\'s Cross',
                                'King\'s Cross', 'King\'s Cross',
                                'King\'s Cross'],
            'Lat': [51.4944, 51.4944, 51.5158, 51.5317, 51.5317,
                    51.5317, 51.5317, 51.5317, 51.5317, 51.5317],
            'Long': [-0.1004, -0.1004, -0.1889, -0.1246, -0.1246,
                     -0.1246, -0.1246, -0.1246, -0.1246, -0.1246],
            'Nearest_Airport': ['London City Airport', 'London City Airport',
                                'Heathrow', 'Stansted', 'Stansted', 'Stansted',
                                'Stansted', 'Stansted', 'Stansted', 'Stansted']
        })

        expected = pd.DataFrame({
            'Postcode': ['SE25 4DZ', 'SE25 4DZ', 'W9 3BQ', 'SW18 1JE',
                         'SW18 1JE', 'N12 0NL', 'N12 0NL', 'N12 0NL',
                         'N12 0NL', 'N12 0NL'],
            'Price': [13.017005, 12.429020, 12.948012, 14.405280, 13.976780,
                      12.295689, 13.158376, 12.729324, 12.362012, 9.210440],
            'Year': [2013, 2012, 2013, 2017, 2019, 2014, 2017, 2019, 2013,
                     2018],
            'Original Price': [450000.0, 249950.0, 420000.0, 1803571.0,
                               1175000.0, 218750.0, 518333.0, 337500.0,
                               233750.0, 10000.0],
            'Postcode_prefix': ['SE', 'SE', 'W', 'SW', 'SW', 'N',
                                'N', 'N', 'N', 'N'],
            'Price Group': ['200k-500k', '200k-500k', '200k-500k',
                            '1.5m-2.5m', '1m-1.5m', '200k-500k',
                            '500k-750k', '200k-500k', '200k-500k', '5k-20k']
        })

        result = asyncio.run(loader._calculate_price_features(data))
        result = result.reset_index(drop=True)
        expected = expected.reset_index(drop=True)
        dtype = pd.CategoricalDtype(categories=loader._PRICE_LABELS,
                                    ordered=True)
        expected['Price Group'] = expected['Price Group'].astype(dtype)
        pd.testing.assert_frame_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
