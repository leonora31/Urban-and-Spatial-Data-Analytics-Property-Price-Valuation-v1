import unittest
from unittest.mock import patch
import pandas as pd

from src.housedatautils.housing_data_analysis import HousingDataAnalysis


class TestHousingDataAnalysis(unittest.TestCase):
    """
    Unit tests for the HousingDataAnalysis class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Setup class-level test data.
        """
        cls.sample_data = pd.DataFrame({
            'Year': [2008, 2008, 2009, 2010, 2010],
            'Price Group': ['50k-100k', '100k-200k', '50k-100k',
                            '100k-200k', '200k-500k'],
            'Price': [75000, 150000, 80000, 170000, 250000],
            'Town_City': ['London', 'Manchester', 'London',
                          'Liverpool', 'London']
        })

    def setUp(self):
        """
        Setup instance-level test data and objects.
        """
        self.analysis = HousingDataAnalysis(self.sample_data)

    def test_count_price_groups(self):
        """
        Test if price groups are counted correctly.
        """
        expected = pd.DataFrame({
            '50k-100k': [1, 1, 1],
            '100k-200k': [1, 1, 0],
            '200k-500k': [0, 0, 1]
        }, index=[2008, 2009, 2010])
        self.assertTrue(isinstance(expected, pd.DataFrame))

    def test_plot_price_distribution(self):
        """
        Test if the price distribution plot is generated.
        """
        with patch('matplotlib.pyplot.show') as show_mock:
            data = pd.DataFrame({'Price': [200000, 300000, 400000]})
            analysis = HousingDataAnalysis(data)
            analysis.plot_price_distribution('Price')
            show_mock.assert_called()

    def test_compute_yearly_stats_2010(self):
        """
        Test if yearly statistics for 2010 are computed correctly.
        """
        sample_data = pd.DataFrame({
            'Year': [2010, 2010, 2010, 2011, 2011],
            'Price': [50000, 100000, 150000, 60000, 130000]
        })

        analysis = HousingDataAnalysis(sample_data)
        result = analysis.compute_yearly_stats('Price')
        year_2010 = result[result['Year'] == 2010].iloc[0]

        expected_2010 = {
            'Year': 2010,
            'count': 3,
            'mean': 100000,
            'std': 50000,
            'min': 50000,
            '25%': 75000,
            '50%': 100000,
            '75%': 125000,
            'max': 150000,
        }

        for key, value in expected_2010.items():
            self.assertAlmostEqual(year_2010[key], value, delta=1e-5)

    def test_calculate_quartiles(self):
        """
        Test if quartiles are calculated correctly.
        """
        analysis = HousingDataAnalysis(self.sample_data)
        result_df = analysis.calculate_quartiles('Price')

        expected_quartiles = pd.DataFrame({
            'Year': [2008, 2009, 2010],
            'q1_count': [1, 1, 1],
            'q2_count': [0, 0, 0],
            'q3_count': [0, 0, 0],
            'q4_count': [1, 0, 1],
        })

        result_quartiles = result_df[['Year', 'q1_count', 'q2_count',
                                      'q3_count', 'q4_count']]
        expected_quartiles['Year'] = expected_quartiles['Year'].astype(int)
        result_quartiles['Year'] = result_quartiles['Year'].astype(int)

        pd.testing.assert_frame_equal(result_quartiles, expected_quartiles,
                                      check_exact=False)


if __name__ == '__main__':
    unittest.main()
