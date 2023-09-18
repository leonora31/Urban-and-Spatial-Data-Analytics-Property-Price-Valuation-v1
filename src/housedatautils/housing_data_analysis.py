"""
HousingDataAnalysis Module
==========================

This module provides tools for analyzing and visualizing housing data.
The main class, HousingDataAnalysis, offers methods to compute statistics,
visualize price distributions, and other useful operations related to housing
data.

Features:
- Compute the number of entries in each price group for each year.
- Filter data to include only frequently occurring postcodes.
- Visualize the distribution of prices using histograms and boxplots.
- Compute yearly statistics for the average price.
- Calculate quartiles for a specified column and visualize them.
- Visualize the number of transfers in each price group for each year.
- Geographic visualization with choropleth plotting.
- Analyze price groups across locations and time.
- Download data and plots for offline use.

Usage Example
-------------
from this_module_name import HousingDataAnalysis

analysis = HousingDataAnalysis(data)
price_group_counts = analysis.count_price_groups()
analysis.show_price_distribution()
statistics = analysis.compute_yearly_stats()
print(statistics)
merged_df = analysis.calculate_quartiles('Average Price')
print(merged_df)
analysis.plot_quartiles('Average Price')
analysis.plot_price_group_counts(price_group_counts)
analysis.plot_price_group_counts(price_group_counts, ['50k-200k', '200k-500k',
'500k-750k', '750k-1m'])
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from matplotlib.ticker import FuncFormatter


class HousingDataAnalysis:
    """
    A class to perform analysis and visualization on housing data.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.selected_cities_data = pd.DataFrame()

    # ------------------------- Analysis Methods -------------------------

    def count_price_groups(self) -> pd.DataFrame:
        """Count the number of entries in each price group for each year."""
        return self.data.groupby(
                ['Year', 'Price Group']
                ).size().unstack()

    def compute_yearly_stats(self, column_name) -> pd.DataFrame:
        """
        Compute yearly statistical metrics for a specified column from 2008 to
        2019.

        This method calculates the standard statistical metrics (count, mean,
        std, min,  25%, 50%, 75%, max) as well as skewness and kurtosis for
        the data in the specified column, grouped by year.

        Parameters:
        - column_name (str): The name of the column in the dataset for which
                            the yearly statistics are to be computed.

        Returns:
        - pd.DataFrame: A DataFrame containing the computed statistics for
                        each year.
                        Each row corresponds to a year and the columns
                        correspond to the statistical metrics.

        Notes:
        - The method assumes the existence of a 'Year' column in the dataset.
        - Only the years from 2008 to 2019 are considered for computation.
        """
        statistics_list = []

        for year in range(2008, 2020):
            category = column_name
            subset = self.data[self.data['Year'] == year][category]
            stats = subset.describe()
            stats['Year'] = year
            stats['Skewness'] = subset.skew()
            stats['Kurtosis'] = subset.kurt()
            statistics_list.append(stats)

        return pd.concat(
            statistics_list,
            axis=1).transpose().reset_index(drop=True)

    def calculate_quartiles(self, column_name: str) -> pd.DataFrame:
        """
        Calculate and return the quartile counts for the specified column,
        merged with its yearly statistics.

        For each year, the method computes the counts of data points within
        each of the quartile ranges (Q1, Q2, Q3, and Q4) for the provided
        column. It then merges these quartile counts with the yearly
        statistics of the same column.

        Parameters:
        - column_name (str): The name of the column in the dataset for which
                            quartiles and statistics are to be computed.

        Returns:
        - pd.DataFrame: A DataFrame that contains the yearly statistical
                        metrics and quartile counts for the given column. Each
                        row corresponds to a year, with columns detailing the
                        statistics and quartile counts.

        Notes:
        - The method assumes the existence of a 'Year' column in the dataset
          to group the data by year.
        - The quartiles are computed based on the quantile method.
        """
        quartiles = []
        for year, grouped in self.data.groupby('Year'):
            q1 = grouped[column_name].quantile(0.25)
            q2 = grouped[column_name].quantile(0.50)
            q3 = grouped[column_name].quantile(0.75)
            quartile_counts = {
                'Year': year,
                'q1_count': grouped[grouped[column_name] <= q1].shape[0],
                'q2_count': grouped[(grouped[column_name] > q1) &
                                    (grouped[column_name] <= q2)].shape[0],
                'q3_count': grouped[(grouped[column_name] > q2) &
                                    (grouped[column_name] <= q3)].shape[0],
                'q4_count': grouped[grouped[column_name] > q3].shape[0],
            }
            quartiles.append(quartile_counts)
        quartiles_df = pd.DataFrame(quartiles)
        statistics = self.compute_yearly_stats(column_name)

        return pd.merge(left=statistics, right=quartiles_df, on='Year')

    # ------------------------- Visualization Methods -------------------------

    def show_price_distribution(self) -> None:
        """
        Display the distribution of property prices using histogram and
        boxplot.

        The method first plots a histogram showing the frequency distribution
        of property prices. Next, it displays a boxplot to visualize the
        central tendency, variability, and potential outliers in the prices.
        Additionally, it prints the skewness and kurtosis values for the
        distribution.

        Histogram:
        - X-axis: property prices.
        - Y-axis: frequency of each price range.

        Boxplot:
        - Y-axis: property prices, showing quartiles and outliers.

        Skewness and kurtosis values provide insights into asymmetry and tail
        behavior of the distribution.

        Notes:
        - Assumes existence of a 'Price' column in the dataset.
        - Uses 'plt' (matplotlib) for plotting; ensure it's imported.

        """
        plt.hist(self.data['Price'], bins=30)
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.show()

        plt.boxplot(self.data['Price'])
        plt.ylabel('Price')
        plt.show()

        print('Skewness:', self.data['Price'].skew())
        print('Kurtosis:', self.data['Price'].kurt())

    def plot_quartiles(self, column_name: str) -> None:
        """
        Plot histograms for each quartile of a specified column per year.

        This method plots histograms showing the distribution of data in each
        quartile for a given column, separated by year.

        Parameters:
        - column_name (str): Column name for which quartiles are calculated.

        Notes:
        - Assumes existence of a 'Year' column in the dataset.
        - Uses 'plt' (matplotlib) for plotting; ensure it's imported.
        """
        quartiles = self.data.groupby('Year')[column_name]
        quartiles = quartiles.quantile([0.25, 0.5, 0.75]).unstack(level=-1)
        years = sorted(self.data['Year'].unique())
        n_years = len(years)
        fig, axs = plt.subplots(n_years, 4, figsize=(20, 5 * n_years))
        for idx, year in enumerate(years):
            data_year = self.data[self.data['Year'] == year]
            quartiles_year = quartiles.loc[year]
            filtered_data = data_year[data_year[column_name]
                                      <= quartiles_year[0.25]]
            axs[idx, 0].hist(filtered_data[column_name], bins=50)
            axs[idx, 0].set_title(f'Q1 - {year}')
            axs[idx, 1].hist(data_year[(data_year[column_name] >
                                        quartiles_year[0.25]) &
                                       (data_year[column_name] <=
                                        quartiles_year[0.5])][column_name],
                             bins=50)
            axs[idx, 1].set_title(f'Q2 - {year}')
            axs[idx, 2].hist(data_year[(data_year[column_name] >
                                        quartiles_year[0.5]) &
                                       (data_year[column_name] <=
                                        quartiles_year[0.75])][column_name],
                             bins=50)
            axs[idx, 2].set_title(f'Q3 - {year}')
            axs[idx, 3].hist(data_year[data_year[column_name] >
                                       quartiles_year[0.75]][column_name],
                             bins=50)
            axs[idx, 3].set_title(f'Q4 - {year}')
        plt.tight_layout()
        plt.show()

    def plot_price_group_counts(self, price_group_counts: pd.DataFrame,
                                groups_to_keep: list = None) -> None:
        """
        Plot the number of transfers for each price group, separated by year.

        Parameters:
        - price_group_counts (pd.DataFrame): Data with group counts per year.
        - groups_to_keep (list, optional): List of groups to retain in the
          plot.

        Notes:
        - Uses 'plt' (matplotlib) for plotting; ensure it's imported.
        """
        if groups_to_keep:
            filtered_counts = price_group_counts[groups_to_keep]
        else:
            filtered_counts = price_group_counts

        # Create a color mapping for all groups
        groups_all = filtered_counts.columns.tolist()
        colors_all = plt.cm.viridis(np.linspace(0, 1, len(groups_all)))

        # Create a dictionary to map group names to simple labels
        label_map = {group: chr(65+i) for i, group in enumerate(groups_all)}

        # Create a 5x4 grid of subplots
        fig, axs = plt.subplots(4, 4, figsize=(20, 15))

        # Flatten the axes so we can easily iterate over them
        axs = axs.flatten()

        for idx, (year, row) in enumerate(filtered_counts.iterrows()):
            # Plot a bar chart for the current year with different colors
            # for each group
            axs[idx].bar([label_map[x] for x in row.index],
                         row.values,
                         color=colors_all)
            axs[idx].set_title(f'Year: {year}')
            axs[idx].set_xlabel('Price Group')
            axs[idx].set_ylabel('Number of Transfers')

        # Create legend in the last subplot
        for i, group in enumerate(groups_all):
            axs[-1].bar(0, 0,
                        color=colors_all[i],
                        label=f'{label_map[group]}: {group}')

        axs[-1].legend()
        axs[-1].axis('off')  # Hide axes

        # Remove the unused subplots
        for idx in range(len(price_group_counts), len(axs) - 1):
            fig.delaxes(axs[idx])

        plt.tight_layout()
        plt.show()

    def plot_price_distribution(self, column):
        """
        Plots the distribution of house prices with mean and median lines.

        Args:
        - house_data (pd.DataFrame): DataFrame containing the house data.

        Returns:
        - None (shows the plot).
        """
        # Calculate the mean and median of the 'Original Price' column
        house_data_mean = self.data[column].mean()
        house_data_median = self.data[column].median()

        # Plot the distribution of 'Original Price'
        sns.displot(data=self.data, x=column)

        # Plot the mean and median as vertical lines on the distribution plot
        plt.axvline(x=house_data_mean, color='blue', label='Mean')
        plt.axvline(x=house_data_median, color='red', linestyle='--',
                    label='Median')

        # Add a legend to the plot
        plt.legend()

        # Display the plot
        plt.show()

    def plot_aggregated_prices(self, year, ax, aggregate_function, label_name,
                               linestyle='-'):
        """
        Plot aggregated house prices based on a provided aggregation function.

        Parameters:
        - year (int): Year for which data is to be aggregated and plotted.
        - ax: Axes object to plot on.
        - aggregate_function (Callable): Function (e.g., 'mean' or 'median')
            to aggregate data.
        - label_name (str): Label for the plotted data.
        - linestyle (str, optional): Style of the line in the plot.
            Default is solid.

        Notes:
        - Assumes 'self.data' contains the data.
        - Expects a 'Year' column in 'self.data' and uses 'plt' for plotting.
        """
        yearly_data = self.data[self.data['Year'] == year].copy()
        aggregated_data = yearly_data.groupby('Month')['Original Price'].agg(
            aggregate_function)

        ax.plot(aggregated_data, label=f'{label_name} {year}',
                linestyle=linestyle, marker='o', markersize=5)

    def plot_house_prices_comparison(self, years=[2019, 2008]):
        """
        Plot a comparison of mean and median house prices for given years.

        Parameters:
        - years (list, optional): List of years to compare.
        Default is [2019, 2008].

        Notes:
        - Uses 'plt' (matplotlib) for plotting.
        - Assumes there's a helper function `plot_aggregated_prices` to
        compute aggregated data.
        """

        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(12, 6))

        # Define the linestyles
        linestyles = ['dashed', '-']

        for year, linestyle in zip(years, linestyles):
            self.plot_aggregated_prices(year, ax, 'median', 'median',
                                        linestyle)
            self.plot_aggregated_prices(year, ax, 'mean', 'mean', linestyle)

        # Set labels, title, grid, and move the legend to the right
        ax.set(xlabel='Month',
               ylabel='Price (£)',
               title="The median and mean housing prices in the UK over" +
                     " months in the provided years\n")

        # Update y-ticks labels to represent values in thousands with 'k'
        y_ticks = ax.get_yticks()
        ax.set_yticklabels(['{:.0f}k'.format(y_val/1000) for y_val in y_ticks])

        # Set x-ticks for each month
        # Assuming 'Month' is a numeric column from 1 to 12
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                            'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)

        plt.grid(True, which='both', axis='both', alpha=0.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Show the plot
        plt.tight_layout()
        plt.show()

    def plot_city_prices(self):
        """
        Plot price changes over time for selected cities.

        Notes:
        - Assumes 'self.selected_cities_data' contains the data.
        - Uses 'pd' (pandas) and a helper function '_create_plotly_figure' and
        '_configure_layout' to generate and customize the plot.
        """
        # Make a copy of the input data to avoid modifying the original -
        # plots moving graph
        df = self.selected_cities_data.copy()

        print(df.head())
        # Convert the 'Date of Transfer' column to a datetime type
        df['Date of Transfer'] = pd.to_datetime(df['Date of Transfer'])

        # Set 'Date of Transfer' as the index for easier time-based operations
        df.set_index('Date of Transfer', inplace=True)

        # Create the plotly figure with all the city traces
        fig = self._create_plotly_figure(df)

        # Configure the layout, annotations, and other aesthetics of the figure
        self._configure_layout(fig, df)

        # Display the resulting plot
        fig.show()

    def plot_cities_data(self):
        """
        Plot the prices for cities in the dataset over the years.

        Notes:
        - Assumes 'self.data' and 'self.selected_cities_data' contain the data.
        - Uses 'sns' (Seaborn), 'plt' (matplotlib) and 'FuncFormatter'
        to format price.
        """
        def format_price(price, _):
            """Helper function to format price for y-axis."""
            return f'{price/1e3:.0f}k'

        cities_df = self.data.copy()
        cities = list(self.selected_cities_data["Town/City"].unique())
        cities_df = cities_df[cities_df["Town_City"].isin(cities)]

        all_years = cities_df['Year'].unique()

        # Plot first graph
        sns.relplot(kind='line', data=cities_df, x='Year', y='Original Price',
                    hue='Town_City', aspect=2.5)
        plt.xticks(rotation=45)
        plt.xticks(all_years)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_price))
        plt.show()

        # Plot second graph
        sns.relplot(kind='line', data=cities_df, x='Year', y='Price',
                    hue='Town_City', aspect=2.5)
        plt.xticks(rotation=45)
        plt.xticks(all_years)
        plt.show()

    # ------------------------- Utility Methods -------------------------
    def filter_frequent_postcodes(self, threshold: int = 8) -> None:
        """
        Filter rows to retain only those postcodes that occur frequently.

        Parameters:
        - threshold (int): Minimum number of occurrences to qualify as
          frequent.
        """
        postcodes_counts = self.data['Postcode'].value_counts()
        frequent_postcodes = postcodes_counts[postcodes_counts >= threshold]
        self.data = self.data[self.data['Postcode'].isin(
            frequent_postcodes.index)]

    @staticmethod
    def remove_cheap_exp_groups(df, cheapest_group: str = "0-5k",
                                expensive_group: str = "100m+"):
        """
        Exclude specified cheap and expensive groups from the data.

        Parameters:
        - df (pd.DataFrame): The input dataframe.
        - cheapest_group (str): The lower price group to remove.
        - expensive_group (str): The upper price group to remove.

        Returns:
        - pd.DataFrame: The dataframe after removing the groups.
        """
        categories = df["Price Group"].cat.categories
        cheapest_groups = categories[:categories.get_loc(cheapest_group)]
        expensive_groups = categories[categories.get_loc(expensive_group):]
        return df[~df["Price Group"].isin(
            list(cheapest_groups) + list(expensive_groups))]

    def get_model_data(self):
        """
        Preprocess the data before modeling.

        Returns:
        - pd.DataFrame: The preprocessed data.
        """
        model_data = self.data.copy()
        model_data['Town_City'] = model_data['Town_City'].factorize(
                                                        )[0].astype('float32')
        model_data['County'] = model_data['County'].factorize(
                                                    )[0].astype('float32')
        encoder = LabelEncoder()
        model_data['Old_New'] = encoder.fit_transform(model_data['Old_New'])
        model_data = pd.get_dummies(model_data, columns=["Property_Type"],
                                    prefix=["Property_Type_is_"])
        model_data.drop(columns=["Postcode", "Original Price",
                                 "Postcode_prefix", "Price Group",
                                 "Month"], inplace=True)
        return model_data

    @staticmethod
    def cache_data(dataframe, cache_path="property.pkl"):
        """
        Cache the data.

        Parameters:
        - dataframe (pd.DataFrame): The data to cache.
        - cache_path (str): The path to the cache file.

        Returns:
        - None: Outputs success or failure message.
        """
        dataframe.to_pickle("preload/" + cache_path)
        print("Pickle data cached correctly to:", cache_path)

    @staticmethod
    def load_cached_data(cache_path="property.pkl"):
        """
        Load the cached data.

        Parameters:
        - cache_path (str): The path to the cache file.

        Returns:
        - data: Loaded data from the cache.
        """
        try:
            with open("preload/" + cache_path, "rb") as file:
                data = pickle.load(file)
            print("Pickle data loaded correctly from:", cache_path)
            return data
        except Exception as e:
            print(f"Failed to load pickle data from {cache_path} due to {e}")

    @staticmethod
    def plot_correlation_heatmap(df):
        """
        Plot a correlation heatmap for the given dataframe.

        Parameters:
        - df (pd.DataFrame): The dataframe to visualize.
        """
        corr_data = df.corr()
        np.fill_diagonal(corr_data.values, 1)
        mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
        plt.figure(figsize=(20, 20))
        sns.heatmap(corr_data, mask=mask, vmax=.3, center=0, square=True,
                    linewidths=.5, annot=True, cmap='RdYlGn')
        plt.show()

    def analyze_price_groups_geo_time(self):
        """
        Analyze price groups over geography and time.

        Returns:
        - pd.DataFrame: Aggregated data on price groups.
        """
        grouped = self.data.groupby(['Postcode', 'Year'])
        stats = grouped['Price'].agg(['mean', 'count'])
        return self.data.merge(stats, on='Postcode')

    def download_plot(self, plt, filename):
        """
        Download the specified plot as an image file.

        Parameters:
        - plt (matplotlib.pyplot): The plot to be saved.
        - filename (str): The filename to save to.
        """
        plt.savefig(filename)
        print(f'Chart saved to {filename}')

    def download_data(self, df, filename):
        """
        Download the given dataframe as a CSV file.

        Parameters:
        - df (pd.DataFrame): The dataframe to save.
        - filename (str): The filename to save to.
        """
        df.to_csv(filename)
        print(f'Data saved to {filename}')

    def show_price_categories(self):
        """
        Display the unique price categories in the data.

        Returns:
        - pd.DataFrame: Unique price categories.
        """
        unique_categories = self.data["Price Group"].unique()
        sorted_categories = sorted(
            unique_categories,
            key=lambda x: list(self.data["Price Group"].cat.categories
                               ).index(x))
        return pd.DataFrame(sorted_categories).T

    def get_selected_cities_to_plot(self, *selected_cities):
        """
        Retrieve selected cities' data for plotting.

        Parameters:
        - *selected_cities: Cities to retrieve.

        Returns:
        - pd.DataFrame: Filtered data for the selected cities.
        """
        most_expensive, middle, cheapest = self._get_three_type_cities()
        if not selected_cities:
            selected_cities = [most_expensive, middle, cheapest]
        selected_cities += ["LONDON"]
        hp = self.load_cached_data("pp_1995_2023.pkl")
        data_to_plot = hp[hp["Town/City"].isin(selected_cities)]
        bins = [0, 5e3, 2e4, 5e4, 1e5, 2e5, 5e5, 7.5e5, 1e6, 1.5e6, 2.5e6, 5e6,
                1e7, 5e7, 1e8, float('inf')]
        labels = ['0-5k', '5k-20k', '20k-50k', '50k-100k', '100k-200k',
                  '200k-500k', '500k-750k', '750k-1m', '1m-1.5m', '1.5m-2.5m',
                  '2.5m-5m', '5m-10m', '10m-50m', '50m-100m', '100m+']
        data_to_plot['Price Group'] = pd.cut(data_to_plot['Price'], bins=bins,
                                             labels=labels)
        data_to_plot = self.remove_cheap_exp_groups(
            data_to_plot,
            cheapest_group="20k-50k",
            expensive_group="1.5m-2.5m")
        self.selected_cities_data = data_to_plot
        return data_to_plot

    # ----------------------- Private Helper Methods -------------------------
    def _create_plotly_figure(self, df):
        """
        Create a Plotly figure of average monthly house prices per city.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing house price data.

        Returns:
        - plotly.graph_objs.Figure: The created Plotly figure.
        """
        fig = go.Figure()
        for city in df['Town/City'].unique():
            monthly_avg = df[df['Town/City'] == city]['Price'].resample(
                'M').mean()
            fig.add_trace(go.Scatter(
                x=monthly_avg.index,
                y=monthly_avg.values,
                mode='lines',
                line=dict(width=1.5),
                opacity=0.8,
                name=f"{city} Monthly Mean House Price"
            ))
        return fig

    def _configure_layout(self, fig, df):
        """
        Configure the layout of a given Plotly figure.

        Parameters:
        - fig (plotly.graph_objs.Figure): The Plotly figure to configure.
        - df (pd.DataFrame): The DataFrame containing house price data.

        Returns:
        - None: Modifies the given figure in place.
        """
        max_price = df['Price'].max()
        min_year = df.index.min().year
        max_year = df.index.max().year
        yearly_ticks = [str(year) + '-01-01' for year in range(min_year,
                                                               max_year+1)]
        shapes = self._get_shapes(max_price)
        annotations = self._get_annotations(max_price)
        fig.update_layout(
            template='gridon',
            title='Average Yearly House Price',
            xaxis_title='Year',
            yaxis_title='Price (£)',
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            legend=dict(x=0.5, y=-.05, orientation='h'),
            shapes=shapes,
            annotations=annotations,
            xaxis=dict(
                tickvals=yearly_ticks,
                ticktext=[str(year) for year in range(min_year, max_year+1)],
                tickangle=-45
            ),
            width=1600,
            height=1200
        )

    def _get_shapes(self, max_price):
        """
        Generate shapes for significant events.

        Parameters:
        - max_price (float): Maximum house price for scaling.

        Returns:
        - list: List of dictionaries defining each shape.
        """
        events = [
            {'x0': '2022-02-23', 'x1': '2022-02-23', 'color': 'red'},
            {'x0': '2020-03-01', 'x1': '2021-07-19', 'color': '#f5cbcc',
             'is_rect': True},
            {'x0': '2016-06-01', 'x1': '2016-06-01', 'color': 'LightSalmon'},
            {'x0': '2007-12-01', 'x1': '2009-06-01', 'color': 'LightSalmon',
             'is_rect': True},
            {'x0': '2001-03-01', 'x1': '2001-11-01', 'color': 'LightSalmon',
             'is_rect': True}
        ]
        return [{
            'type': 'rect' if event.get('is_rect') else 'line',
            'x0': event['x0'],
            'x1': event['x1'],
            'y0': 0,
            'y1': max_price * 1.2,
            'line': {'color': event['color'], 'dash': 'dashdot'},
            'fillcolor': event.get('color') if event.get('is_rect') else None,
            'opacity': 0.5 if event.get('is_rect') else 1,
            'layer': 'below' if event.get('is_rect') else None,
            'line_width': 0 if event.get('is_rect') else 1
        } for event in events]

    def _get_annotations(self, max_price):
        """
        Generate annotations for significant events.

        Parameters:
        - max_price (float): Maximum house price for scaling.

        Returns:
        - list: List of dictionaries defining each annotation.
        """
        labels = [
            ("The Great Recession", '2007-12-01'),
            ("Brexit Vote", '2016-06-01'),
            ("Dot-Com Bubble Recession", '2001-03-01'),
            ("Covid-19", '2020-03-01'),
            ("Russian Invasion", '2022-02-23')
        ]
        return [dict(text=label, x=date, y=max_price*1.2, showarrow=True)
                for label, date in labels]

    def _get_three_type_cities(self):
        """
        Identify three types of cities based on their average yearly price:
        most expensive, middle-priced, and cheapest (from the filtered list)

        Returns:
        - tuple: Names of the most expensive, middle-priced, and cheapest city.
        """
        yearly_avg_prices = self.data.groupby(
            ['Town_City', 'Year']
        )['Original Price'].mean().reset_index()
        city_prices = yearly_avg_prices.groupby(
            'Town_City'
            )['Original Price'].mean().reset_index()
        city_prices.columns = ['City', 'Average Yearly Price']
        threshold_5_percent = city_prices['Average Yearly Price'].quantile(
            0.05)
        filtered_cities = city_prices[city_prices['Average Yearly Price'] >
                                      threshold_5_percent]
        sorted_cities = filtered_cities.sort_values(by='Average Yearly Price',
                                                    ascending=False
                                                    ).reset_index(drop=True)
        most_expensive = sorted_cities.iloc[0]
        cheapest = sorted_cities.iloc[-1]
        middle_value = (most_expensive['Average Yearly Price'] +
                        cheapest['Average Yearly Price']) / 2 - 1
        sorted_cities['Distance to Middle'] = abs(
            sorted_cities['Average Yearly Price'] - middle_value)
        middle = sorted_cities.loc[sorted_cities['Distance to Middle'].idxmin(
        )]
        return most_expensive['City'], middle['City'], cheapest['City']
