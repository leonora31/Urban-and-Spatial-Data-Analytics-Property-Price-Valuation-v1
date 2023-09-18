"""
PropertyPriceMapPlotter

This module contains the PropertyPriceMapPlotter class to visualize property
price prediction differences on a map.


The key features include:

- Plotting price difference data on a map using colour encoding
- Overlaying geographical boundary zones
- Customizable plotting options like text labels, colourbars etc
- Saving generated plots as image files
- Logging and configurable parameters

Usage

Import the PropertyPriceMapPlotter class:

from property_price_map_plotter import PropertyPriceMapPlotter

Create plotter instance:

plotter = PropertyPriceMapPlotter(df, zones_df, options)

where:

- df is a DataFrame with price diff info and geometries
- zones_df contains zone geometries
- options like title, sizes etc

Generate plot:

plotter.plot(plot_text=True, plot_bars=True)

Saves plot images and outputs interactive plot.

Customization

The plot can be customized by:

- Passing diff options to the plot() method
- Configuring class parameters like colours and styles
- Subclassing and overriding methods

Implementation

The main steps are:

- Data preprocessing
- Plotting price diff geometries
- Adding zone boundaries
- Colorbars, text labels etc
- Saving results and images

Uses Geopandas, Pandas, Matplotlib and Plotly under the hood.
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import geopandas as gpd
import logging

from matplotlib.colors import (LinearSegmentedColormap, TwoSlopeNorm)
from matplotlib.ticker import StrMethodFormatter
from .colourize_predictions_dataset import ColourizePredictionsDataset
from .logging_config import setup_logging


class ColourbarParams:
    """
    A utility class to handle parameters for creating colourbars.

    Attributes:
    ----------
    cmap : Colormap
        A colormap instance or registered colormap name.
    norm : Normalise
        Normalise object which scales data, typically into the interval [0, 1].
    label : str
        Label for the colourbar.
    ax : Axis
        The Axis instance in which the colourbar will be drawn.

    Methods:
    --------
    _create_ticks(n_ticks: int) -> list:
        Private method to create the tick values for the colourbar.
    create_colourbar(fmt: Formatter) -> Colourbar:
        Generate and return a colourbar.
    """

    def __init__(self, cmap, vmin, vmax, vcenter, label, ax):
        self.cmap = cmap
        self.norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        self.label = label
        self.ax = ax

    def _create_ticks(self, n_ticks: int) -> list:
        """
        Private method to generate the tick values for the colourbar.

        Parameters:
        ----------
        n_ticks : int
            The number of ticks to generate.

        Returns:
        --------
        list:
            List of tick values.
        """

        tick_step = (self.norm.vmax - self.norm.vmin) / (n_ticks - 1)
        return [self.norm.vmin + i * tick_step for i in range(n_ticks)]

    def create_colourbar(self, fmt):
        """
        Generate a colourbar.

        Parameters:
        ----------
        fmt :
            The format of the colourbar ticks.

        Returns:
        --------
        Colorbar:
            An instance of the created colorbar.
        """
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=self.cmap,
                                                  norm=self.norm),
                            cax=self.ax, orientation='horizontal', format=fmt)
        cbar.set_label(self.label)
        cbar.ax.tick_params(labelsize=10)
        ticks = self._create_ticks(10)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([fmt.format(x=t) for t in ticks])
        return cbar


class PropertyPriceMapPlotter:
    """
    Class to visualize property price data on geographical maps.

    Attributes:
    ----------
    - (Several attributes related to property data and map configurations)

    Methods:
    --------
    plot_data_on_map(ax: Axes, edges: bool, edges_size: float) -> None:
        Plot the property price data on the map.
    plot_zones_on_map(ax: Axes) -> None:
        Plot the major and detailed zones on the map.
    plot(plot_text: bool, plot_bars: bool, plot_edges: bool, percentage: int)
        -> None:
        Main function to plot the property price data on the map along with
        zones and annotations.
    _save_plot_images() -> None:
        Save the generated plot images to the specified directories.
    _plot_colourbars(axes: list, percentage: int) -> None:
        Plot the colourbars on the map.
    _compute_fontsize(area: float, min_font: int, max_font: int) -> float:
        Compute and return the font size based on the area.
    _annotate_postcodes(ax: Axes) -> None:
        Annotate the postcode areas on the map.
    _annotate_single_postcode(ax: Axes, row: DataFrame row) -> None:
        Annotate a single postcode on the map.
    _save_results() -> None:
        Save the results of the property price data analysis to an Excel file.
    _create_results_df() -> DataFrame:
        Create a DataFrame of the results for saving.
    _save_to_excel(data: DataFrame, file_path: str, sheet_name: str) -> None:
        Utility function to save data to an Excel file.
    _get_existing_data(file_path: str) -> dict:
        Retrieve existing data from an Excel file.
    _update_or_add_data(dfs: dict, data: DataFrame, sheet_name: str) -> None:
        Update or add new data to the existing dataset.
    _write_to_excel(dfs: dict, file_path: str) -> None:
        Write the data to the Excel file.
    """

    def __init__(self, df: pd.DataFrame, major_zones_df: pd.DataFrame = None,
                 detailed_zones_df: pd.DataFrame = None,
                 sizes: tuple = (20, 10), edges_size: float = 0.2,
                 area: str = "UK", model: str = "DT", tuned: bool = False,
                 trained_on: str = "UK", accuracy: float = 0.0):
        setup_logging()
        self._logger = logging.getLogger(__name__)
        self._logger.info("PPMapPlotter initialized")
        self.df = df
        self.major_zones_df = major_zones_df
        self.detailed_zones_df = detailed_zones_df
        self.zone_colours = {
            'E': 'darkred', 'EC': 'darkblue', 'N': 'darkgreen',
            'NW': 'slategray', 'SE': 'purple', 'SW': 'darkorange',
            'W': 'sienna', 'WC': 'darkmagenta'
        }
        self.sizes = sizes
        if area == "LONDON":
            edges_size = 0.05
        elif area == "UK":
            edges_size = 0.005
        else:
            edges_size = 0.2
        self.edges_size = edges_size
        self.tuned = tuned
        self.area = area
        self.model_name = model
        self.trained_on = trained_on
        self.accuracy = accuracy
        self.tuned_status = "01" if not tuned else "02_Tuned"
        self.file_name = "_".join([
            self.tuned_status, self.model_name, "Trained_on",
            trained_on, area
        ])
        self.title = " ".join([
            area, "trained on", self.trained_on+":",
            f"{accuracy:.2f}%"
        ])

    def _plot_data_on_map(self, ax: plt.Axes, edges: bool = False,
                          edges_size: float = 0.2):
        """
        Plot property price data on the provided map axes based on its
        relevance.

        Args:
            ax (plt.Axes): The Matplotlib axes on which to plot the data.
            edges (bool, optional): If True, edges will be plotted around each
            area. Defaults to False.
            edges_size (float, optional): Size of the edges if they are
            plotted. Defaults to 0.2.

        This method uses three possible relevance levels: Unknown, No, Yes.
        """
        for relevance, cmap in zip(['Unknown', 'No', 'Yes'],
                                   ['interpolated_colour']*3):
            sub_df = self.df[self.df['Relevant'] == relevance]
            colours = sub_df[cmap].tolist()
            args = (ax, colours) if not edges else (ax, colours, 'black',
                                                    edges_size)
            sub_df.plot(ax=args[0], facecolor=args[1],
                        edgecolor=args[2] if len(args) > 2 else None,
                        linewidth=args[3] if len(args) > 2 else None)
        ax.set_aspect('equal')

    def _plot_zones_on_map(self, ax: plt.Axes) -> None:
        """
        Plot zones on the provided map axes.

        Args:
            ax (plt.Axes): The Matplotlib axes on which to plot the zones.

        The method distinguishes between major and detailed zones and plots
        both on the map.
        """
        if (self.major_zones_df is not None and
                self.detailed_zones_df is not None):
            # Major zones
            for zone_name, zone_data in self.major_zones_df.iterrows():
                current_colour = self.zone_colours.get(zone_name, 'black')
                g = gpd.GeoSeries(zone_data['geometry'].boundary)
                g.plot(ax=ax, color=current_colour, linewidth=1.5)
                centroid = zone_data['geometry'].centroid
                ax.text(
                    centroid.x, centroid.y, zone_name,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=26, color=current_colour, weight='bold'
                )
            # Detailed zones
            for zone_name, zone_data in self.detailed_zones_df.iterrows():
                label_colour = self.zone_colours.get(zone_name[:1], 'black')
                gpd.GeoSeries(zone_data['geometry']).boundary.plot(
                    ax=ax, color='black', linewidth=1
                )
                centroid = zone_data['geometry'].centroid
                ax.text(
                    centroid.x, centroid.y, zone_name,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8, color=label_colour, weight='bold'
                )

    def plot(self, plot_text: bool = False, plot_bars: bool = False,
             plot_edges: bool = False, percentage: int = 10):
        """
        Plot the property price difference map with optional annotations,
        colourbars, and edges.

        Args:
            plot_text (bool, optional): If True, postcodes are annotated on the
            map. Defaults to False.
            plot_bars (bool, optional): If True, colourbars are plotted below
            the map. Defaults to False.
            plot_edges (bool, optional): If True, edges are plotted around each
            area. Defaults to False.
            percentage (int, optional): Percentage value used for creating
            colourbars. Defaults to 10.

        This method generates a complete visualization based on the user's
        requirements.
        """
        if plot_bars:
            fig, axes = plt.subplots(
                4, 1, figsize=self.sizes,
                gridspec_kw={'height_ratios': [0.85, 0.05, 0.05, 0.05]}
                )
            ax1, ax2, ax3, ax4 = axes
        else:
            fig, ax1 = plt.subplots(figsize=self.sizes)

        self._plot_data_on_map(ax1, plot_edges, self.edges_size)
        self._plot_zones_on_map(ax1)

        if plot_bars:
            self._plot_colourbars([ax2, ax3, ax4], percentage)

        if plot_text:
            self._annotate_postcodes(ax1)

        title_str = f"Property Price Difference Map in {self.title}"
        ax1.set_title(title_str)
        plt.tight_layout()

        self._save_plot_images()
        self._save_results()
        plt.show()

    def _save_plot_images(self):
        """
        Save the current plot to disk in multiple formats and qualities.

        The plots are saved in 4K regular quality, high DPI quality, and vector
        format.
        The directory structure is based on the region and whether the model is
        tuned.
        """
        self._logger.info("Saving images...")

        dpi = 300
        tuned_folder = "Tuned" if self.tuned else "Not tuned"
        base_path = f"../results/Regions/Images_{self.area}/{tuned_folder}"
        dpi_scale = 3840 / plt.gcf().get_size_inches()[0]
        reg_path = f"{base_path}/Regular Quality/plot_4k_{self.file_name}.png"
        hr_path = f"{base_path}/High Quality/plot_4k_{self.file_name}_new.png"
        vec_path = f"{base_path}/Vector Quality/plot_4k_{self.file_name}.svg"

        plt.savefig(reg_path, dpi=dpi_scale)
        plt.savefig(hr_path, dpi=dpi)
        plt.savefig(vec_path, format='svg', transparent=True)

    def _plot_colourbars(self, axes, percentage: int):
        """
        Plot colourbars for the map visualization based on the data's relevance.

        Args:
            axes (list): List of matplotlib axes to plot colourbars.
            percentage (int): Percentage value to determine the range and
            division of colourbars.

        Generates three colourbars: one for relevant price deviations, and two
        for irrelevant price deviations.
        """

        ax2, ax3, ax4 = axes

        n_ticks = 10
        dec = int(percentage % n_ticks != 0)
        fmt_str = f'{{x:.{dec}f}}%'
        fmt = StrMethodFormatter(fmt_str)

        colourbar_data = [
            ColourbarParams(
                cmap=LinearSegmentedColormap.from_list("Relevant",
                                                       ["#113300", "#d6ff00"]),
                vmin=min(0, self.df.loc[self.df['Relevant'] == 'Yes',
                                        'Difference_Abs'].min()),
                vmax=percentage,
                vcenter=percentage/2,
                label='Relevant Price Deviations',
                ax=ax2
            ),
            ColourbarParams(
                cmap=LinearSegmentedColormap.from_list("Irrelevant",
                                                       ["#d6ff00", "#ffa100"]),
                vmin=min(percentage, self.df.loc[self.df['Relevant'] == 'No',
                                                 'Difference_Abs'].min()),
                vmax=percentage+10,
                vcenter=percentage+5,
                label='Irrelevant Price Deviations',
                ax=ax3
            ),
            ColourbarParams(
                cmap=LinearSegmentedColormap.from_list("Irrelevant",
                                                       ["#ffa100", "#ff0000"]),
                vmin=percentage+10,
                vmax=self.df.loc[self.df['Relevant'] == 'No',
                                 'Difference_Abs'].max(),
                vcenter=(2*percentage+10+self.df.loc[
                    self.df['Relevant'] == 'No', 'Difference_Abs'
                    ].max())/2,
                label='Irrelevant Price Deviations',
                ax=ax4
            )
        ]

        for colourbar_param in colourbar_data:
            colourbar_param.create_colourbar(fmt)

    def _compute_fontsize(self, area, min_font=4, max_font=8):
        """
        Compute font size for annotations based on area.

        Args:
            area (float): Area value to compute font size.
            min_font (int, optional): Minimum font size. Defaults to 4.
            max_font (int, optional): Maximum font size. Defaults to 8.

        Returns:
            float: Computed font size.

        The computed font size is normalized based on the minimum and maximum
        area values from the dataframe.
        """
        normalized_area = (
            area - self.df['area'].min()
        ) / (self.df['area'].max() - self.df['area'].min())
        return min_font + normalized_area * (max_font - min_font)

    def _annotate_postcodes(self, ax: plt.Axes):
        """
        Annotate postcodes on the provided axes for each row in the dataframe.

        Args:
            ax (plt.Axes): Matplotlib axes to place the annotations.

        Iterates through each row of the dataframe and adds an annotation if
        the 'Difference' column has a valid value.
        """
        for idx, row in self.df.iterrows():
            if pd.notna(row['Difference']):
                self._annotate_single_postcode(ax, row)

    def _annotate_single_postcode(self, ax, row):
        """
        Annotate a single postcode on the provided axes.

        Args:
            ax (plt.Axes): Matplotlib axes to place the annotation.
            row (pd.Series): A single row from the dataframe with data for the
            postcode, geometry, and other relevant columns.

        Places an annotation at the centroid of the geometry for the given
        postcode with its difference and difference price.
        """
        centroid = row['geometry'].centroid
        fontsize = self._compute_fontsize(row['area'])
        annotation_text = (
            f"{row['Postcode']}\n{row['Difference']}\n"
            f"{row['Difference_price']:,.0f}"
        )
        ax.annotate(
            text=annotation_text,
            xy=(centroid.x, centroid.y),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=fontsize * 0.5,
            color='black',
            weight='bold'
        )

    def _save_results(self) -> None:
        """
        Save the results data to an Excel file.

        The function determines the name of the file based on the 'tuned'
        status of the model and then saves the generated results DataFrame to
        an Excel file under the specified path.
        """
        self._logger.info("Saving results...")
        tuned_name = "tuned" if self.tuned else "not_tuned"
        file_path = f"../results/results_{tuned_name}.xlsx"
        results_df = self._create_results_df()
        self._save_to_excel(results_df, file_path, self.area)

    def _create_results_df(self) -> pd.DataFrame:
        """
        Create a DataFrame for the model's results.

        Returns:
            pd.DataFrame: A DataFrame containing the model's name, the dataset
            it was trained on, and its accuracy for a 10% prediction.
        """
        return pd.DataFrame({
            'Model': [self.model_name],
            'Trained on': [self.trained_on],
            'Accuracy 10% Pred': [self.accuracy]
        })

    def _save_to_excel(self, data: pd.DataFrame, file_path: str,
                       sheet_name: str):
        """
        Save the provided DataFrame to an Excel file.

        Args:
            data (pd.DataFrame): DataFrame to be saved.
            file_path (str): Path of the Excel file to save the data.
            sheet_name (str): Name of the sheet in the Excel file to save the
            data.

        Saves the data to the specified Excel file. If the file already exists,
        it updates the existing data with the new data for the specified sheet
        name.
        """
        dfs = self._get_existing_data(file_path)
        self._update_or_add_data(dfs, data, sheet_name)
        self._write_to_excel(dfs, file_path)

    def _get_existing_data(self, file_path: str) -> dict:
        """
        Retrieve existing data from an Excel file.

        Args:
            file_path (str): Path of the Excel file to read data from.

        Returns:
            dict: A dictionary where keys are sheet names and values are
            DataFrames containing the data from the respective sheet.
        """
        if os.path.exists(file_path):
            with pd.ExcelFile(file_path) as xls:
                return {sh: pd.read_excel(xls, sh) for sh in xls.sheet_names}
        return {}

    def _update_or_add_data(self, dfs: dict, data: pd.DataFrame,
                            sheet_name: str) -> None:
        """
        Update existing data with new data or add new data.

        Args:
            dfs (dict): Existing data where keys are sheet names and values are
                        DataFrames.
            data (pd.DataFrame): New data to be updated or added.
            sheet_name (str): Name of the sheet in which data should be
                              updated or added.

        Updates the data in the specified sheet name if it already exists.
        Otherwise, adds the new data as a new sheet.
        """
        if sheet_name in dfs:
            mask = (
                (dfs[sheet_name]['Model'] == data['Model'].iloc[0]) &
                (dfs[sheet_name]['Trained on'] == data['Trained on'].iloc[0])
            )
            dfs[sheet_name] = dfs[sheet_name].loc[~mask]
            dfs[sheet_name] = dfs[sheet_name].append(data, ignore_index=True)
        else:
            dfs[sheet_name] = data

    def _write_to_excel(self, dfs: dict, file_path: str) -> None:
        """
        Write data to an Excel file.

        Args:
            dfs (dict): Data to be written where keys are sheet names and
            values are DataFrames.
            file_path (str): Path of the Excel file to save the data.

        Saves the provided data to the specified Excel file.
        """
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sh, df in dfs.items():
                df.to_excel(writer, sheet_name=sh, index=False)
