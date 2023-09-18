import unittest
import os


class TestPropertyPriceMapPlotter(unittest.TestCase):

    def test_save_results(self):
        self.assertTrue(os.path.exists('./results/results_not_tuned.xlsx'))


if __name__ == '__main__':
    unittest.main()
