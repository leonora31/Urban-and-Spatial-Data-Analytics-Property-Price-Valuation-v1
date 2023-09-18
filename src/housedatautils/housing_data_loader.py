"""HousingDataLoader

This module provides the HousingDataLoader class to help load and preprocess
housing data for analysis and modeling.

The key features include:

- Loading data from multiple CSV files
- Merging different datasets into a single DataFrame
- Data cleaning and preprocessing
- Feature engineering for prices
- Filtering to certain years and postcodes
- Caching merged data to avoid reloading

# Usage

Load the module and instantiate the HousingDataLoader.

```python
from housing_data_loader import HousingDataLoader

loader = HousingDataLoader()
Call the load method to load and preprocess data.

python

Copy code

data = await loader.load()
This will return a cleaned Pandas DataFrame ready for analysis.

The class handles:

Reading CSV files
Merging multiple tables
Handling missing data
Adding derived columns
It provides a simple unified interface to load ready-to-use housing data.

Customization
The class allows customization of:

Raw data file locations
Loading method
Data preprocessing steps
This makes the class adaptable to new data sources or use cases.

"""

from .base_data_loader import BaseDataLoader, LoaderType
from typing import Optional, Callable
import numpy as np
import pandas as pd
from pandarallel import pandarallel


class HousingDataLoader(BaseDataLoader):
    """
    HousingDataLoader: A class to efficiently load and preprocess
    housing data based on various conditions.

    Usage:
    loader = HousingDataLoader()
    data = await loader.load_inner_london()
    """

    _NEAREST_STATION_DISTANCE_THRESHOLD = 3
    _NEAREST_AIRPORT_DISTANCE_THRESHOLD = 20
    _NEAREST_PARK_DISTANCE_THRESHOLD = 3000
    _PRICE_BINS = [0, 5e3, 2e4, 5e4, 1e5, 2e5, 5e5, 7.5e5, 1e6, 1.5e6, 2.5e6,
                   5e6, 1e7, 5e7, 1e8, float('inf')]
    _PRICE_LABELS = ['0-5k', '5k-20k', '20k-50k', '50k-100k', '100k-200k',
                     '200k-500k', '500k-750k', '750k-1m', '1m-1.5m',
                     '1.5m-2.5m', '2.5m-5m', '5m-10m', '10m-50m', '50m-100m',
                     '100m+']

    def __init__(self,
                 loader_type: LoaderType = LoaderType.HOUSING,
                 read_method: Optional[Callable] = None
                 ):
        super().__init__(loader_type=loader_type,
                         read_method=read_method)

        pandarallel.initialize()

    async def _rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Rename columns in the DataFrame.

        This renames two columns in the provided DataFrame:

        - 'Postcode_x' is renamed to simply 'Postcode'
        - The column with long name starting with 'Average_number...'
        is shortened to 'Avg num of parks'

        Args:
            data (pd.DataFrame): Input DataFrame to rename columns for.

        Returns:
            pd.DataFrame: DataFrame with renamed columns.

        """

        data = data.rename(columns={'Postcode_x': 'Postcode'})
        return data.rename(columns={"Average_number_of_parks_or_public_" +
                                    "gardens_within_1_000_m_radius":
                                    "Avg num of parks"})

    async def _merge_datasets(self, properties: pd.DataFrame,
                              properties_trains: pd.DataFrame,
                              properties_airports: pd.DataFrame,
                              properties_green: pd.DataFrame,
                              inflation_rate: pd.DataFrame,
                              bank_rate: pd.DataFrame,
                              postcode_mapping: pd.DataFrame,
                              crimes: pd.DataFrame
                              ) -> pd.DataFrame:
        """Merge multiple data sources into a single DataFrame.

        This merges the given input DataFrames together:

        - properties
        - properties_trains
        - properties_airports
        - properties_green
        - postcode_mapping
        - bank_rate
        - inflation_rate
        - crimes

        The merging is done sequentially by chaining multiple calls to merge
        helper methods like `_merge_properties_and_trains`.

        The order of merging is optimized to allow merging on common columns.

        Args:
            properties (pd.DataFrame): Base DataFrame
            properties_trains (pd.DataFrame): Train stations data
            properties_airports (pd.DataFrame): Airports data
            properties_green (pd.DataFrame): Green areas data
            postcode_mapping (pd.DataFrame): Postcode mappings
            bank_rate (pd.DataFrame): Bank rate data
            inflation_rate (pd.DataFrame): Inflation rate data
            crimes (pd.DataFrame): Crimes data

        Returns:
            pd.DataFrame: Merged DataFrame

        """
        self._logger.info("Merging propertires and train data.")
        data = await self._merge_properties_and_trains(properties,
                                                       properties_trains)

        self._logger.info("Merging data with airport data.")
        data = await self._merge_with_airports(data, properties_airports)

        self._logger.info("Merging data with postcode encoding data.")
        data = await self._merge_with_postcode_mapping(data, postcode_mapping)

        self._logger.info("Merging data with airport data.")
        data = await self._merge_with_green_areas(data, properties_green)

        self._logger.info("Merging data with bank rate data.")
        data = await self._merge_with_bank_rate(data, bank_rate)

        self._logger.info("Merging data with inflation rate data.")
        data = await self._merge_with_inflation_rate(data, inflation_rate)

        self._logger.info("Merging data with crime data.")
        return await self._merge_with_crimes(data, crimes)

    async def _merge_properties_and_trains(self, properties,
                                           properties_trains):
        """Merge properties data with properties_trains data.

        Args:
            properties (pd.DataFrame): Properties data.
            properties_trains (pd.DataFrame): Properties trains station data.

        Returns:
            pd.DataFrame: Merged data on 'Postcode' column.
        """
        return pd.merge(properties, properties_trains, on='Postcode')

    async def _merge_with_airports(self, data, properties_airports):
        """Merge data with airports data.

        Args:
            data (pd.DataFrame): Input data to merge.
            properties_airports (pd.DataFrame): Airports data.

        Returns:
            pd.DataFrame: Merged data on 'Postcode' column.
        """
        return pd.merge(data, properties_airports, on='Postcode')

    async def _merge_with_postcode_mapping(self, data, postcode_mapping):
        """Merge data with postcode mapping info.

        Postcode mapping is reset index to avoid issues.
        Unnamed column is dropped from postcode mapping data.

        Args:
            data (pd.DataFrame): Input data to merge.
            postcode_mapping (pd.DataFrame): Postcode mapping data.

        Returns:
            pd.DataFrame: Merged data on 'Postcode' column.

        """

        postcode_mapping = postcode_mapping.reset_index(drop=True)
        data = pd.merge(data, postcode_mapping, on='Postcode')
        data.drop(columns=['Unnamed: 0'], inplace=True)
        return data

    async def _merge_with_green_areas(self, data, properties_green):
        """Merge data with green areas info.

        Green areas data requires postcode formatting before merge.
        Postcodes are standardized by removing whitespace.
        Data is merged on 'Postcode_no_space' column.

        Args:
            data (pd.DataFrame): Input data to merge.
            properties_green (pd.DataFrame): Green areas data.

        Returns:
            pd.DataFrame: Merged data.
        """

        postcodes_no_space = properties_green['Postcode'].parallel_apply(
            lambda x: x.replace(' ', '')
        )
        properties_green['Postcode_no_space'] = postcodes_no_space
        data['Postcode_no_space'] = data['Postcode'].parallel_apply(
            lambda x: x.replace(' ', '')
        )
        return pd.merge(data, properties_green, on='Postcode_no_space',
                        how='left')

    async def _merge_with_bank_rate(self, data, bank_rate):
        """Merge data with bank rate information.

        The bank rate data is merged into the input data DataFrame
        on the 'Year' column.

        A left join is used so that all rows from the input data are
        retained, with the bank rate columns added where available.

        Args:
            data (pd.DataFrame): Input data to merge with bank rate.
            bank_rate (pd.DataFrame): DataFrame containing bank rate data.

        Returns:
            pd.DataFrame: Input DataFrame merged with bank rate data.

        """
        return pd.merge(data, bank_rate, on='Year', how='left')

    async def _merge_with_inflation_rate(self, data, inflation_rate):
        """Merge data with inflation rate information.

        The inflation rate data is merged into the input data DataFrame
        on the 'Year' column.

        A left join is used so that all rows from the input data are
        retained, with the inflation rate columns added where available.

        Args:
            data (pd.DataFrame): Input data to merge with inflation rate.
            inflation_rate (pd.DataFrame): DataFrame containing inflation data.

        Returns:
            pd.DataFrame: Input DataFrame merged with inflation rate data.

        """
        return pd.merge(data, inflation_rate, on='Year', how='left')

    async def _merge_with_crimes(self, data, crimes):
        """Merge data with crime stats.

        Crimes data is preprocessed before merging:
        - Rename latitude/longitude columns
        - Merge on year, latitude and longitude
        - Fill NA crimes values with 0

        Args:
        - data (pd.DataFrame): Input data to merge.
        - crimes (pd.DataFrame): Crimes data.

        Returns:
        - pd.DataFrame: Merged data.
        """

        crimes.rename(columns={"Latitude": "Lat", "Longitude": "Long"},
                      inplace=True)

        data = pd.merge(data, crimes, on=['Year', 'Lat', 'Long'], how='left')
        data['Number_of_crimes'].fillna(0, inplace=True)

        return data

    async def _apply_threshold(self, series: pd.Series,
                               threshold: int) -> pd.Series:
        """Apply a threshold filter to a Pandas Series.

        Values below the threshold are set to 1, others are set to 0.
        Applied in parallel using Series.parallel_apply.

        Args:
            series (pd.Series): Input numeric series.
            threshold (int): Threshold value.

        Returns:
            pd.Series: Series with 0/1 values based on threshold.
        """
        return series.parallel_apply(lambda x: 1 if x < threshold else 0)

    async def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data and handle missing values.

        Apply thresholded filters for distance columns.
        Drop redundant distance columns after filtering.
        Drop other unnecessary columns like temporary ID columns.
        Remove rows with missing values.

        Args:
            data (pd.DataFrame): Raw data to clean.

        Returns:
            pd.DataFrame: Cleaned data.
        """
        distance_thresholds = {
            'Nearest Station <3 km': (
                'Nearest_Station_Distance',
                self._NEAREST_STATION_DISTANCE_THRESHOLD
            ),
            'Nearest Park <3 km': (
                'Average_distance_to_nearest_park_or_public_garden__m_',
                self._NEAREST_PARK_DISTANCE_THRESHOLD
            ),
            'Nearest Airport <20 km': (
                'Nearest_Airport_Distance',
                self._NEAREST_AIRPORT_DISTANCE_THRESHOLD
            )
        }

        # Apply threshold checks
        for new_col, (old_col, threshold) in distance_thresholds.items():
            data[new_col] = await self._apply_threshold(data[old_col],
                                                        threshold)

        # Drop old columns
        old_columns_to_drop = [old_col for old_col,
                               _ in distance_thresholds.values()]
        data.drop(columns=old_columns_to_drop, inplace=True)

        # Drop additional columns
        additional_columns_to_drop = ['Postcode_no_space', 'Postcode_y']
        data.drop(columns=additional_columns_to_drop, inplace=True)

        # Drop rows containing NaN values
        data.dropna(inplace=True)

        return data

    async def _calculate_price_features(self,
                                        data: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived price related features.

        This sequentially executes multiple data transformations:

        - Store original price
        - Extract postcode prefix
        - Drop unnecessary columns
        - Calculate average price
        - Bin prices into groups

        Args:
            data (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data with new price features
        """
        data = await self._store_original_price(data)
        data = await self._extract_postcode_prefix(data)
        data = await self._drop_unnecessary_columns(data)
        data = await self._calculate_average_price(data)
        return await self._categorize_price_into_groups(data)

    async def _store_original_price(self, data: pd.DataFrame) -> pd.DataFrame:
        """Store the original 'Price' column as 'Original Price'.

        This creates a separate column to preserve the original prices
        before any transformations are applied.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data with original prices stored
        """
        data['Original Price'] = data['Price']
        return data

    async def _extract_postcode_prefix(self,
                                       data: pd.DataFrame) -> pd.DataFrame:
        """Extract the prefix from the postcode as a new column.

        For postcodes with a space, the prefix is the substring before the
        space.
        For postcodes without space, the first 1 or 2 chars are taken as the
        prefix.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data with postcode prefix extracted
        """
        data['Postcode_prefix'] = data['Postcode'].parallel_apply(
            lambda x: x[0] if x[1].isdigit() else x[:2]
        )
        return data

    async def _drop_unnecessary_columns(self,
                                        data: pd.DataFrame) -> pd.DataFrame:
        """Drop columns not needed for price modeling.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data with unnecessary columns dropped
        """
        return data.drop(columns=['Nearest_Station', 'Lat', 'Long',
                                  'Nearest_Airport'])

    async def _calculate_average_price(self,
                                       data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the average price per postcode per year.

        Averages the 'Price' column grouped by year and postcode.
        Applies log transform to normalize the price distribution.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data with average price per group.
        """
        grouped_cols = ['Year', 'Postcode']
        avg_price_by_group = data.groupby(grouped_cols)['Price']
        data['Original Price'] = avg_price_by_group.transform('mean')
        data['Price'] = np.log1p(data['Original Price'])
        return data.drop_duplicates(subset=['Original Price', 'Postcode',
                                            'Year'])

    async def _categorize_price_into_groups(self,
                                            data: pd.DataFrame
                                            ) -> pd.DataFrame:
        """Categorize prices into predefined bins.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data with binned price groups.
        """

        data['Price Group'] = pd.cut(data['Original Price'],
                                     bins=self._PRICE_BINS,
                                     labels=self._PRICE_LABELS)
        return data

    async def _update_crime_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Update data with crime count totals for each postcode area.

        This adds a 'Crimes_Buffer' column which contains the total number of
        crimes in the broader postcode area for each row.

        The steps are:

        1. Add Postcode_no_last_letter: Postcode without the last letter.

        2. Group by Postcode_no_last_letter and Year.
        Sum the 'Number_of_crimes' to get total crimes for each area.

        3. Join the summed crime data back to the original dataset.

        This provides crime context for each postcode based on the broader
        postcode area.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Input data with crime buffer data added.
        """
        data = await self._add_postcode_no_last_letter(data)
        grouped_data = await self._group_and_aggregate_crime_data(data)
        return await self._merge_with_original_data(data, grouped_data)

    async def _add_postcode_no_last_letter(self,
                                           data: pd.DataFrame
                                           ) -> pd.DataFrame:
        """Add postcode column without final letter.

        This adds a 'Postcode_no_last_letter' column containing the postcode


        This is used to aggregate crimes by the broader postcode area.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data with new postcode column added.
        """
        data['Postcode_no_last_letter'] = data['Postcode'].str[:-1]
        return data

    async def _group_and_aggregate_crime_data(self,
                                              data: pd.DataFrame
                                              ) -> pd.DataFrame:
        """Group by postcode and year, aggregating crimes.

        Groups the data by 'Postcode_no_last_letter' and 'Year', summing the
        'Number_of_crimes' to get the total crimes for each postcode area and
        year.

        Sorted descending by the 'Crimes_Buffer' total.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data grouped and aggregated by postcode and year.
        """
        return data.groupby(['Postcode_no_last_letter', 'Year']).agg(
                Crimes_Buffer=('Number_of_crimes', 'sum')
               ).reset_index().sort_values(by='Crimes_Buffer', ascending=False)

    async def _merge_with_original_data(self,
                                        data: pd.DataFrame,
                                        grouped_data: pd.DataFrame
                                        ) -> pd.DataFrame:
        """Merge crime data back to original data.

        Joins the grouped and aggregated crime data back to the original data
        using the 'Postcode_no_last_letter' and 'Year' columns.

        Removes the temporary column after join.

        Args:
            data (pd.DataFrame): Original input data
            grouped_data (pd.DataFrame): Crime data grouped

        Returns:
            pd.DataFrame: Original data merged with crime data.
        """
        merged_data = pd.merge(data, grouped_data,
                               left_on=['Postcode_no_last_letter', 'Year'],
                               right_on=['Postcode_no_last_letter', 'Year'],
                               how='left').sort_values(
                                   by='Crimes_Buffer', ascending=False
                                )
        merged_data.drop(columns=['Postcode_no_last_letter'], inplace=True)
        return merged_data

    async def _filter_years(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data to specified years and postcodes.

        Filters the data to:
        - Postcodes existing in the 2019 data
        - Years between 2008 and 2019 inclusive

        Args:
            data (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data filtered by year and postcodes.
        """

        postcodes_2019 = await self._get_postcodes_from_year(data, 2019)
        data = await self._filter_data_by_years(data, 2008, 2019)
        return await self._filter_data_by_postcodes(data, postcodes_2019)

    async def _get_postcodes_from_year(self, data: pd.DataFrame,
                                       year: int) -> pd.Series:
        """Extract unique postcodes for a given year.

        Filters the data to only the specified year and returns the unique
        postcode values.

        Args:
            data (pd.DataFrame): Input data containing 'Year' and 'Postcode'
            year (int): Year to extract postcodes for

        Returns:
            pd.Series: Unique postcode values for the given year.
        """
        return data[data['Year'] == year]['Postcode'].unique()

    async def _filter_data_by_years(self, data: pd.DataFrame, start_year: int,
                                    end_year: int) -> pd.DataFrame:
        """Filter data to retain only given range of years.

        Filters the data to only rows where the 'Year' column is between the
        start_year and end_year values inclusive.

        Args:
            data (pd.DataFrame): Input data
            start_year (int): Minimum year value to keep
            end_year (int): Maximum year value to keep

        Returns:
            pd.DataFrame: Data filtered to the specified range of years.
        """
        return data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]

    async def _filter_data_by_postcodes(self, data: pd.DataFrame,
                                        postcodes: pd.Series) -> pd.DataFrame:
        """Filter data to only include the given postcodes.

        Args:
            data (pd.DataFrame): Input data
            postcodes (pd.Series): Postcodes to retain

        Returns:
            pd.DataFrame: Data filtered to only the specified postcodes.
        """
        return data[data['Postcode'].isin(postcodes)]

    async def _process_data(self):
        """Load, merge and preprocess housing data.

        This executes the main data processing pipeline:

        1. Load data from files and merge datasets.
        2. Perform cleaning and preprocessing.
        3. Calculate derived features.
        4. Assign to the `data` attribute.

        The merged and processed DataFrame is returned.

        Returns:
            pd.DataFrame: Loaded, merged and preprocessed housing data.
        """
        data = await self._load_and_merge_data()
        data = await self._perform_data_preprocessing(data)
        self.data = data
        return self.data

    async def _load_and_merge_data(self):
        """Load and merge raw data if not already cached.

        Loads the raw data files by calling the BaseDataLoader and merges them:

        - If data is cached, return cached version
        - Otherwise load raw data from files
        - Merge individual dataframes
        - Return merged DataFrame

        Returns:
            pd.DataFrame: Raw merged data from files.
        """
        if not self._data_cached or self.data is None:
            self._logger.info("Loading datasets.")
            dataframes = await self._load_data("data_paths")
            self._logger.info("Merging datasets.")
            return await self._merge_datasets(*dataframes)
        return self.data

    async def _perform_data_preprocessing(self,
                                          data: pd.DataFrame) -> pd.DataFrame:
        """Execute data preprocessing pipeline.

        Sequentially performs:

        - Cleaning
        - Column renaming
        - Filtering
        - Feature engineering

        Args:
            data (pd.DataFrame): Input data to preprocess.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        self._logger.info("Cleaning data.")
        data = await self._clean_data(data)

        self._logger.info("Renaming columns.")
        data = await self._rename_columns(data)

        self._logger.info("Filtering data between 2008-2019.")
        data = await self._filter_years(data)

        self._logger.info("Updating crime data.")
        data = await self._update_crime_data(data)

        self._logger.info("Calculating price features.")
        return await self._calculate_price_features(data)
