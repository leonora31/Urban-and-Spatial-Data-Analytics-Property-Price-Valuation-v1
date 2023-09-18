"""
ColourizePredictionsDataset Module

This module provides the `ColourizePredictionsDataset` class, designed to
process, compare, and visualize the actual and predicted values of a dataset.
It also integrates geospatial data for enhanced visualizations, offering
methods for merging, preparing, and processing dataframes to compute relevant
insights and visualize them.

The module utilizes `geopandas` for geospatial data processing, `pandas` for
general data manipulation, and custom utilities like `GeoDataJSONLoader` for
loading geospatial data and `setup_logging` for structured logging.

Classes:
--------
- `ColourizePredictionsDataset`: The primary class for data processing and
visualization.

Dependencies:
-------------
- geopandas: For handling geospatial data.
- pandas: For data manipulation tasks.
- numpy: For numerical computations.
- logging: For logging information.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import logging

from pandas import Series, DataFrame
from typing import Union
from .logging_config import setup_logging


class ColourizePredictionsDataset:

    """
    A class designed to process and visualize the comparison between actual
    and predicted values for a given dataset, incorporating geospatial data for
    enhanced visualization.

    """
    def __init__(self, y_test: Series, predictions: list,
                 postcode_mapping: DataFrame,
                 temp_data: DataFrame,
                 geo_data: DataFrame,
                 percentage: int = 10):
        setup_logging()
        self._logger = logging.getLogger(__name__)
        self._logger.info("Colourised Module initialized")
        self.y_test_series = pd.Series(y_test, name="y_test")
        self.predictions = predictions
        self.postcode_mapping = postcode_mapping
        self.temp_data = temp_data
        self.percentage = percentage
        self._geo_data = geo_data
        self._df = None
        self.postcode_counts = None

        self._prepare_dataframe()
        self._logger.info("Dataframe prepared")
        self._merge_geo_data()

    def _merge_geo_data(self) -> None:
        """
        Merges the main dataframe with geospatial data for visualization.

        This method performs the following steps:
        1. Extracts a subset of the main dataframe focusing on columns
        necessary for geospatial merging.
        2. Converts the 'Difference' column values to float type for
        computation.
        3. Merges this subset with the preloaded geospatial data on the
        'Postcode' column.
        4. Calls another method to assign default values to the merged
        dataframe.

        Logging:
        - Logs the start and end of the geospatial merging process.
        - Notifies when metrics for areas are set.

        Attributes Affected:
        - Modifies the internal `_merged` attribute of the class.
        """

        self._logger.info("Merging geodata...")

        # Subset the dataframe
        df_subset = self._df[['Postcode', 'Difference',
                              'Difference_price', 'Relevant']]
        df_subset['Difference_Float'] = (df_subset['Difference']
                                         .str.rstrip('%').astype('float'))

        # Update the merged attribute of the class
        self._merged = pd.merge(self._geo_data, df_subset, how='left',
                                left_on='Postcode', right_on='Postcode')
        self._logger.info("Metrics for area are to be settled")
        self._assign_defaults_to_merged_data()
        self._logger.info("Assigning metrics colours")
        self._assign_colours()

    def _prepare_dataframe(self) -> None:
        """ Prepares the primary dataframe by comparing predictions and
        processing the data. This method structures the dataframe, cleans it,
        calculates major zones, and processes postcode efficiency.
        """
        self._construct_comparison_dataframe()
        self._clean_dataframe()
        self._calculate_major_zones()
        self._process_postcode_efficiency()

    def _construct_comparison_dataframe(self) -> None:
        """
        Constructs a dataframe for comparison of actual and predicted values.

        This method merges actual values from test series, predicted values,
        and other supplementary data to create a comparative dataframe.
        The differences between actual and predicted values are computed and
        the relevance of each prediction is determined.

        Attributes Affected:
        - Initializes and populates the internal `_df` attribute of the class.
        """
        self._compare_predictions(self.y_test_series, self.predictions,
                                  self.postcode_mapping, self.temp_data,
                                  self.percentage)

        self._logger.info("We are going to compare predictions now")

    def _clean_dataframe(self) -> None:
        """
        Cleans and formats the primary comparison dataframe.

        This method carries out several dataframe operations such as:
        - Removing '%' from the Difference column and converting its type.
        - Computing the absolute difference and sorting based on it.
        - Formatting the Difference column.
        - Dropping unnecessary columns and NaN values.

        Attributes Affected:
        - Modifies the internal `_df` attribute of the class.
        """

        self._df['Difference'] = self._df['Difference'].str.replace(
            '%', '').astype(float)
        self._df['Abs_Difference'] = self._df['Difference'].abs()
        self._df = self._df.sort_values('Abs_Difference', ascending=False)
        self._df['Difference'] = self._df['Difference'].astype(str) + '%'
        self._df = self._df.drop('Abs_Difference', axis=1).dropna()

    def _calculate_major_zones(self) -> None:
        """
        Extracts major zones from postcodes in the dataframe.

        The major zone corresponds to the first part of a UK postcode (before
        the space).
        This method computes and adds a new 'Major_Zone' column to the
        dataframe.

        Attributes Affected:
        - Adds a 'Major_Zone' column to the internal `_df` attribute.
        """

        self._df['Major_Zone'] = self._df['Postcode'].str.split(' ').str[0]

    def _process_postcode_efficiency(self) -> None:
        """
        Calculates postcode efficiency based on prediction relevance.

        This method analyzes predictions by major zone and determines their
        relevance.
        It constructs a new dataframe (`postcode_counts`) which quantifies
        the number of relevant and irrelevant predictions per major zone,
        and computes an efficiency score for each.

        Attributes Affected:
        - Initializes and populates the `postcode_counts` attribute of the
        class.
        """

        # Count occurrences based on Major_Zone and Relevant columns
        counts = (self._df.groupby(['Major_Zone', 'Relevant'])
                          .size()
                          .unstack(fill_value=0))

        # Rename columns for clarity
        counts.columns = ['Irrelevant', 'Relevant']

        # Calculate Total and Efficiency columns
        counts['Total'] = counts['Relevant'] + counts['Irrelevant']
        counts['Efficiency'] = counts['Relevant'] / counts['Total']

        # Sort by Efficiency and assign to postcode_counts attribute
        self.postcode_counts = counts.sort_values(by="Efficiency")

    def _assign_defaults_to_merged_data(self) -> None:
        """
        Assigns default values to the merged geospatial data.

        This method handles missing data in the merged dataframe. It assigns
        default values to NaN entries in specific columns, computes the
        absolute difference for each entry, and calculates the area for
        geospatial entries.

        Attributes Affected:
        - Modifies the internal `_merged` attribute of the class.
        """

        self._merged['Difference_Float'].fillna(0, inplace=True)
        self._merged['Relevant'].fillna('Unknown', inplace=True)
        self._merged["Difference_Abs"] = abs(self._merged["Difference_Float"])
        self._merged['area'] = self._merged['geometry'].area

    def _calculate_geospatial_metrics(self) -> None:
        """
        Calculates various geospatial metrics for the merged data.

        This method:
        - Sets the geometry for the dataframe.
        - Calculates the distance of each entry to a global centroid.
        - Filters out outliers based on interquartile range (IQR) calculations.
        """

        self._merged = self._merged.set_geometry('geometry')
        global_centroid = self._merged.geometry.unary_union.centroid
        distances = self._merged.geometry.centroid
        self._merged['distance_to_center'] = distances.distance(
            global_centroid)

        # Filter out outliers based on IQR
        Q1, Q3 = self._merged['distance_to_center'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        self._merged = self._merged[self._merged['distance_to_center'] <=
                                    (Q3 + 1.5 * IQR)]

    def _assign_colours(self) -> None:
        """
        Assigns colour categorization and interpolated colours to the data.

        This method:
        - Categorizes each entry based on its relevance and difference
        percentage.
        - Calculates a normalized difference for colour interpolation.
        - Assigns an interpolated colour to each entry based on its categorized
        group.
        """

        self._merged['colour_group'] = self._merged.apply(
            self._assign_colour_group, axis=1)

        normalized_diff = self._merged.groupby('colour_group')[
            'Difference_Abs'].transform(self._groupwise_normalize)
        self._merged['Normalized_Difference'] = normalized_diff
        self._merged['interpolated_colour'] = self._merged.apply(
            lambda row: self._colour_by_group(row['Normalized_Difference'],
                                              row['colour_group']), axis=1)

    @staticmethod
    def _assign_colour_group(row: pd.Series,
                             threshold_percentage: int = 10) -> int:
        """
        Categorize an entry based on its relevance and difference percentage.

        Parameters:
        - row (pd.Series): A row from the dataframe.
        - threshold_percentage (int): The threshold percentage for
        categorization.

        Returns:
        - int: A colour group categorization.
        """

        if row['Relevant'] == "Unknown":
            return 0

        if row['Relevant'] == "Yes":
            return 1
        if (row['Relevant'] == "No" and threshold_percentage <=
           row['Difference_Abs'] <= threshold_percentage + 10):
            return 2

        return 3

    @staticmethod
    def _groupwise_normalize(group: pd.Series) -> Union[float, pd.Series]:
        """
        Normalize a group of values between 0 and 1.

        If the maximum and minimum value in the group are the same,
        it returns 0.

        Parameters:
        - group (pd.Series): A group of values.

        Returns:
        - Union[float, pd.Series]: A normalized value or a series of normalized
        values.
        """
        min_val, max_val = group.min(), group.max()
        if max_val == min_val:
            return 0.0
        return (group - min_val) / (max_val - min_val)

    @staticmethod
    def _colour_by_group(value: float, group: int) -> str:
        """
        Determines the colour for a data entry based on its group.

        Parameters:
        -----------
        value (float): Normalized difference or position within group.
        group (int): The group the data entry belongs to.

        Returns:
        --------
        str
            The hex representation of the colour corresponding to the
            data's group and position.
        """

        if group == 0:  # Unknown
            return "#eeeeee"

        if group == 1:  # Relevant ("Yes")
            start_colour, end_colour = [17, 51, 0], [214, 255, 0]
        elif group == 2:  # Irrelevant subset 1 ("No")
            start_colour, end_colour = [214, 255, 0], [255, 161, 0]
        elif group == 3:  # Irrelevant subset 2 ("No")
            start_colour, end_colour = [255, 161, 0], [255, 0, 0]

        # Linearly interpolate between the colours
        r = (start_colour[0] + value * (end_colour[0] - start_colour[0]))
        g = (start_colour[1] + value * (end_colour[1] - start_colour[1]))
        b = (start_colour[2] + value * (end_colour[2] - start_colour[2]))

        # Convert the resulting RGB values back to hex
        return '#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))

    def _merge_data(self, y_test: Series, predictions: list,
                    postcode_data: DataFrame,
                    temp_data: DataFrame) -> DataFrame:
        """
        Merge dataframes for predictions comparison.

        Parameters:
        -----------
        y_test (Series): Actual values.
        predictions (list): Model predictions.
        postcode_data (DataFrame): Unique postcode mappings.
        temp_data (DataFrame): Temporary postcode data.

        Returns:
        --------
        DataFrame: Unified dataframe with merged data.
        """
        postcode_data = postcode_data.drop_duplicates(subset="Postcode")
        merged_data = pd.merge(y_test, temp_data[["Postcode"]],
                               left_index=True, right_index=True)

        self._df = pd.DataFrame()
        self._df['Postcode'] = merged_data[["Postcode"]].reset_index(drop=True)
        self._df['Actual'] = y_test.reset_index(drop=True)
        self._df['Predicted'] = predictions

        return self._df

    def _calculate_differences(self, df: DataFrame,
                               percentage: int) -> DataFrame:
        """
        Calculate difference and relevance in predictions.

        Parameters:
        -----------
        df (DataFrame): Data with 'Predicted' and 'Actual' columns.
        percentage (int): Threshold for determining relevance.

        Returns:
        --------
        DataFrame: Updated dataframe with difference calculations.
        """
        df['Difference'] = ((df['Predicted'] - df['Actual'])
                            / df['Actual']) * 100
        df['Difference_price'] = df['Difference'] * df['Actual'] / 100
        df['Relevant'] = np.where(
            (df['Difference'] <= percentage) &
            (df['Difference'] >= -percentage), 'Yes', 'No'
        )

        return df

    def _format_dataframe(self, df: DataFrame) -> DataFrame:
        """
        Format specific columns of the dataframe for better clarity.

        Parameters:
        -----------
        df (DataFrame): Data with columns to format.

        Returns:
        --------
        DataFrame: Formatted dataframe.
        """
        for col in ['Actual', 'Predicted']:
            df[col] = df[col].apply(lambda x: "{:,}".format(int(x)))
        df['Difference'] = df['Difference'].apply(
            lambda x: "{:.0f}%".format(x)
        )

        return df

    def _compare_predictions(self, y_test: Series, predictions: list,
                             postcode_data: DataFrame, temp_data: DataFrame,
                             percentage: int) -> DataFrame:
        """
        Compare predicted values against actuals and derive insights.

        Parameters:
        -----------
        y_test (Series): Actual values.
        predictions (list): Model predictions.
        postcode_data (DataFrame): Postcode mappings.
        temp_data (DataFrame): Temporary postcode data.
        percentage (int): Threshold for relevance.

        Returns:
        --------
        DataFrame: Computed dataframe with comparisons.
        """
        self._df = self._merge_data(y_test, predictions, postcode_data,
                                    temp_data)
        self._df = self._calculate_differences(self._df, percentage)
        self._df = self._format_dataframe(self._df)

        return self._df

    def _count_relevant(self) -> int:
        """
        Counts the number of predictions that are deemed relevant.

        Returns:
        --------
        int
            Number of relevant predictions.
        """
        num_relevant = self._df['Relevant'].value_counts().get('Yes', 0)
        total_predictions = len(self._df)
        percent_relevant = (num_relevant / total_predictions) * 100

        print(f"Number of decent predictions: {num_relevant}")
        print(f"Total predictions: {total_predictions}")
        print(f"Percentage of decent predictions: {percent_relevant:.2f}%")
        self.accuracy = percent_relevant
        return num_relevant
