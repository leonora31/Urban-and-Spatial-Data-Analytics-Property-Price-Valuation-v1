import unittest
import pandas as pd
import geopandas as gpd
import pandas as pd
import numpy as np

from unittest.mock import patch, Mock
from src.housedatautils import ColourizePredictionsDataset


class TestColourizePredictionsDataset(unittest.TestCase):

    def setUp(self):

        temp_data = {
            'Price': [6488533, 6488481, 6488482, 6488483],
            'Postcode': ['NE34 6QJ', 'NE34 6AX', 'NE34 6AX', 'NE34 6AX'],
            'Old_New': [False, False, False, False],
            'Town_City': ['SOUTH SHIELDS'] * 4,
            'County': ['TYNE AND WEAR'] * 4,
            'Year': [2017, 2012, 2015, 2017],
            'Month': [11, 4, 5, 8],
            'Property_Type': ['F', 'F', 'D', 'F'],
            'Postcode_encoded': [624841, 624712, 624712, 624712],
            'Avg num of parks': [3.705882, 4.547619, 4.547619, 4.547619],
            'Rate': [0.50, 1.00, 1.00, 0.50],
            'Inflation rate': [2.6] * 4,
            'Number_of_crimes': [0.0] * 4,
            'Nearest Station <3 km': [0] * 4,
            'Nearest Park <3 km': [1] * 4,
            'Nearest Airport <20 km': [1] * 4,
            'Crimes_Buffer': [0.0] * 4,
            'Original Price': [111500.000000, 60333.333333, 368333.333333, 
                               48950.000000],
            'Postcode_prefix': ['NE'] * 4,
            'Price Group': ['100k-200k', '50k-100k', '200k-500k', '20k-50k']
        }

        # Assigning custom index to temp_data
        custom_index = [6488478, 6488536, 6488266, 6489053]
        self.temp_data = pd.DataFrame(temp_data, index=custom_index)

        postcode_mapping = {
            'Postcode': ['NE34 6QJ', 'NE34 6AX'],
            'Postcode_encoded': [0.0, 1.0]
        }

        self.postcodes = pd.DataFrame(postcode_mapping)

        self.y_test = pd.Series({
            6488478: 590001.000000,
            6488536: 120001.000000,
            6488266: 94001.000000,
            6489053: 78001.000000
        })

        self.predictions = np.array([
            277109.8823235,
            109080.38458008,
            74767.31238315,
            193291.00511465
        ])

        # Assigning custom index to temp_data
        custom_index = [6488478, 6488536, 6488266, 6489053]
        self.temp_data = pd.DataFrame(self.temp_data, index=custom_index)

        geodata = {
            'mapit_code': [
                'NE346QJ',
                'NE346AX',
                'NE346AX',
                'NE346AX'
            ],
            'Postcode': [
                'NE34 6QJ',
                'NE34 6AX',
                'NE34 6AX',
                'NE34 6AX'
            ],
            'Postcode_prefix': [
                'NE',
                'NE',
                'NE',
                'NE'
            ]
        }

        geodata_df = gpd.GeoDataFrame(geodata)
        self.geo_data = gpd.GeoDataFrame(geodata)

        # Read the geojson file
        geo_df = gpd.read_file("./tests/sample.geojson")

        # Calculate number of entries in geodata_df
        num_entries = len(geodata_df)

        # Pull that many entries from the geometry column of geo_df
        selected_geometries = geo_df['geometry'].iloc[:num_entries]

        # Assign these geometries to the geodata_df
        self.geo_data['geometry'] = selected_geometries.values

    def test_merge_data(self):
        dataset = ColourizePredictionsDataset(self.y_test,
                                              self.predictions,
                                              self.postcodes,
                                              self.temp_data,
                                              self.geo_data)

        # Set a name for the y_test Series
        self.y_test.name = "y_test_name"

        result = dataset._merge_data(self.y_test, self.predictions,
                                     self.postcodes, self.temp_data)

        data = {
            'Postcode': ['NE34 6QJ', 'NE34 6AX', 'NE34 6AX', 'NE34 6AX'],
            'Actual': [590001.0, 120001.0, 94001.0, 78001.0],
            'Predicted': [277109.882324, 109080.384580, 74767.312383,
                          193291.005115]
        }

        expected_df = pd.DataFrame(data)

        pd.testing.assert_frame_equal(result, expected_df)

    def test_calculate_differences(self):
        # Setup
        data_input = {
            'Postcode': ['NE34 6QJ', 'NE34 6AX', 'NE34 6AX', 'NE34 6AX'],
            'Actual': [590001.0, 120001.0, 94001.0, 78001.0],
            'Predicted': [277109.882324, 109080.384580, 74767.312383,
                          193291.005115]
        }
        input_df = pd.DataFrame(data_input)

        dataset = ColourizePredictionsDataset(self.y_test,
                                              self.predictions,
                                              self.postcodes,
                                              self.temp_data,
                                              self.geo_data)
        # Act
        result_df = dataset._calculate_differences(input_df, percentage=10)

        # Expected dataframe
        data_expected = {
            'Postcode': ['NE34 6QJ', 'NE34 6AX', 'NE34 6AX', 'NE34 6AX'],
            'Actual': [590001.0, 120001.0, 94001.0, 78001.0],
            'Predicted': [277109.882324, 109080.384580, 74767.312383,
                          193291.005115],
            'Difference': [-53.032303, -9.100437, -20.461152, 147.805804],
            'Difference_price': [-312891.117676, -10920.615420, -19233.687617,
                                 115290.005115],
            'Relevant': ['No', 'Yes', 'No', 'No']
        }
        expected_df = pd.DataFrame(data_expected)

        # Assertion
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_process_postcode_efficiency(self):
        # Setup
        data_input = {
            'Major_Zone': ['NE34', 'NE34', 'NE34', 'NE34'],
            'Relevant': ['Yes', 'No', 'No', 'No']
        }
        input_df = pd.DataFrame(data_input)

        dataset = ColourizePredictionsDataset(self.y_test,
                                              self.predictions,
                                              self.postcodes,
                                              self.temp_data,
                                              self.geo_data)
        dataset._df = input_df  # Assign sample data to _df attribute

        # Act
        dataset._process_postcode_efficiency()

        # Expected result
        data_expected = {
            'Irrelevant': [3],
            'Relevant': [1],
            'Total': [4],
            'Efficiency': [0.25]
        }
        expected_df = pd.DataFrame(data_expected, index=['NE34'])
        expected_df.index.name = 'Major_Zone'  # set index name to 'Major_Zone'

        # Assertion
        pd.testing.assert_frame_equal(dataset.postcode_counts, expected_df)

    def test_assign_colours(self):
        # Setup
        data_input = {
            'mapit_code': ['NE346QJ', 'NE346AX'],
            'Postcode': ['NE34 6QJ', 'NE34 6AX'],
            'geometry': ["POLYGON ((-2.55106 53.38458, -2.55109 53.38455...))",
                        "MULTIPOLYGON (((-2.58702 53.38507, -2.58650 53...))"],
            'Postcode_prefix': ['NE', 'NE'],
            'Difference': ['-53.0%', '148.0%'],
            'Difference_price': [-312891.117676, 115290.005115],
            'Relevant': ['No', 'No'],
            'Difference_Float': [-53.0, 148.0],
            'Difference_Abs': [53.0, 148.0],
            'area': [0.000001, 0.000010]
        }
        input_df = pd.DataFrame(data_input)

        dataset = ColourizePredictionsDataset(self.y_test,
                                              self.predictions,
                                              self.postcodes,
                                              self.temp_data,
                                              self.geo_data)
        dataset._merged = input_df  # Assign sample data to _merged attribute

        # Act
        dataset._assign_colours()

        # Expected dataframe columns
        data_expected = {
            'colour_group': [3, 3],
            'Normalized_Difference': [0.0, 1.0],
            'interpolated_colour': ['#ffa100', '#ff0000']
        }
        expected_df = pd.DataFrame(data_expected)

        # Assertion
        result_df = dataset._merged[['colour_group', 'Normalized_Difference',
                                     'interpolated_colour']]
        pd.testing.assert_frame_equal(result_df, expected_df)




if __name__ == "__main__":
    unittest.main()
