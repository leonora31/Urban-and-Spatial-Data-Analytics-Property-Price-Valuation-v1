"""
model_visualization_tools
-------------------------
Provides tools for analyzing and visualizing model results:

- `ModelAnalysis`: Processes and analyzes model's results.
- `RegionAccuracyPlotter`: Visualizes accuracy plots for regions.
- `DataFrameStyler`: Styles and displays dataframes.

Classes:
--------
ModelAnalysis: Processes and analyzes model's results.
RegionAccuracyPlotter: Visualizes accuracy for regions.
DataFrameStyler: Styles and displays dataframes.

Usage:
------
from model_visualization_tools import ModelAnalysis,
                                      RegionAccuracyPlotter,
                                      DataFrameStyler
"""

import copy
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.ndimage import zoom
from matplotlib.colors import LinearSegmentedColormap
from IPython.core.display import display, HTML

from .colourize_predictions_dataset import ColourizePredictionsDataset
from .pp_map_plotter import PropertyPriceMapPlotter


class ModelAnalysis:
    """
    Processes and analyzes model's results.

    Methods:
    --------
    process_all() -> None:
        Processes all steps.
    """
    def __init__(self):
        # Suppress warnings temporarily
        warnings.filterwarnings("ignore")

        # Set regions to consider
        self.regions = ["UK", "LONDON", "ESHER", "WOODSTOCK", "SOUTH SHIELDS"]

        # Set initial paths
        self.results_path = "../results/results_{}.xlsx"
        self.summary_path = "../results/summary_results.xlsx"

        # Placeholder for dataframes
        self.df_not_tuned = None
        self.df_tuned = None

    def _reshape_region_data(self, df, region, tuned, trained_on):
        """
        Reshape data for a given region based on tuning and training data.

        Parameters:
        -----------
        df : DataFrame
            Input data.
        region : str
            Region of interest.
        tuned : bool
            If True, considers tuned models.
        trained_on : str
            Criteria for 'Trained on'.

        Returns:
        --------
        reshaped_df : DataFrame
            Reshaped data in the required format.
        """
        df_filtered = self._filter_data_by_training(df, trained_on)
        model_columns = ["DT", "LR", "XGBoost", "LightGBM", "RNN"]

        reshaped = {"Region": region}
        for model in model_columns:
            reshaped[model] = self._get_accuracy_for_model(df_filtered,
                                                           model)
        reshaped["Trained on"] = trained_on
        reshaped["Tuned"] = "Yes" if tuned else "No"

        return pd.DataFrame([reshaped])

    def _filter_data_by_training(self, df, trained_on):
        """Filter data by the 'Trained on' criteria."""
        return df[df['Trained on'] == trained_on]

    def _get_accuracy_for_model(self, df, model):
        """Retrieve accuracy for a specific model, if available."""
        if model in df['Model'].values:
            return df[df['Model'] == model]['Accuracy 10% Pred'].values[0]
        return None

    def _create_summary_tables(self, tuned=False):
        """
        Create summary tables for model results based on tuning and
        training data.

        Parameters:
        -----------
        tuned : bool, optional
            If True, considers tuned models. Default is False.

        Notes:
        ------
        This method doesn't return anything but writes the processed
        dataframes to an Excel file.
        """
        tuned_path = "not_tuned" if not tuned else "tuned"
        data_path = f"../results/results_{tuned_path}.xlsx"
        regions = ["UK", "LONDON", "ESHER", "WOODSTOCK", "SOUTH SHIELDS"]
        results_df_uk = pd.DataFrame()
        results_df_regions = pd.DataFrame()

        for region in regions:
            df = pd.read_excel(data_path, sheet_name=region)
            region_data_uk = self._reshape_region_data(df, region, tuned, 'UK')
            results_df_uk = pd.concat([results_df_uk, region_data_uk])

            if region != "UK":
                region_data_region = self._reshape_region_data(
                    df, region, tuned, region)
                results_df_regions = pd.concat([results_df_regions,
                                                region_data_region])

        with pd.ExcelWriter("../results/summary_results.xlsx",
                            mode='a' if tuned else 'w') as writer:
            sheet_name_uk = "Trained on UK" + \
                            f"{'Tuned' if tuned else 'Not Tuned'}"
            sheet_name_regions = "Trained on Regions" + \
                                 f" {'Tuned' if tuned else 'Not Tuned'}"
            results_df_uk.to_excel(writer, sheet_name=sheet_name_uk,
                                   index=False)
            results_df_regions.to_excel(writer,
                                        sheet_name=sheet_name_regions,
                                        index=False)

    def _combine_and_sort_data(self, tuned=False):
        """
        Combine and sort model results based on tuning and region.

        Parameters:
        -----------
        tuned : bool, optional
            If True, considers tuned models. Default is False.

        Returns:
        --------
        combined_results : DataFrame
            Combined and sorted results.
        """
        tuned_path = "not_tuned" if not tuned else "tuned"
        data_path = f"../results/results_{tuned_path}.xlsx"
        regions = ["UK", "LONDON", "ESHER", "WOODSTOCK", "SOUTH SHIELDS"]
        combined_results = pd.DataFrame()

        for region in regions:
            df = pd.read_excel(data_path, sheet_name=region)
            combined_results = pd.concat(
                [combined_results,
                 self._reshape_region_data(df, region, tuned, 'UK')])
            if region != "UK":
                combined_results = pd.concat(
                    [combined_results,
                     self._reshape_region_data(df, region, tuned, region)])
        model_columns = ["DT", "LR", "XGBoost", "LightGBM", "RNN"]
        max_model = combined_results[model_columns].mean().idxmax()
        combined_results = combined_results.sort_values(by=max_model,
                                                        ascending=False)

        return combined_results

    def _save_combined_results(self):
        """
        Private method to save the combined results of tuned and non-tuned
        models into an Excel file.

        This method does not return anything but generates or appends to an
        Excel file in the specified directory.
        """
        combined_not_tuned = self._combine_and_sort_data(tuned=False)
        combined_tuned = self._combine_and_sort_data(tuned=True)

        with pd.ExcelWriter("../results/summary_results.xlsx",
                            mode='a') as writer:
            combined_not_tuned.to_excel(writer,
                                        sheet_name="Combined Not Tuned",
                                        index=False)
            combined_tuned.to_excel(writer,
                                    sheet_name="Combined Tuned",
                                    index=False)

    def load_summary_data(self):
        """
        Load summary data from a pre-defined Excel file and store it in
        class attributes.

        Attributes Updated:
        -------------------
        df_not_tuned : DataFrame
            Contains the summary data for non-tuned models.
        df_tuned : DataFrame
            Contains the summary data for tuned models.
        """
        # Path to the Excel file
        summary_path = "../results/summary_results.xlsx"

        # Load the sheets into dataframes
        self.df_not_tuned = pd.read_excel(summary_path,
                                          sheet_name="Combined Not Tuned")
        self.df_tuned = pd.read_excel(summary_path,
                                      sheet_name="Combined Tuned")

    def process_all(self):
        """
        Run the entire processing pipeline to:
        1. Create summary tables for both tuned and non-tuned models.
        2. Save combined results of these models.
        3. Load the processed summary data into class attributes.

        This method serves as a convenience method to execute the entire
        processing in a sequence.
        """
        self._create_summary_tables(tuned=False)
        self._create_summary_tables(tuned=True)
        self._save_combined_results()
        self.load_summary_data()


class RegionAccuracyPlotter:
    """
    Visualizes accuracy plots for regions.

    Methods:
    --------
    plot_region_accuracy(region: str, tuned: bool=False) -> None:
        Plots accuracy for a region.
    """
    def __init__(self):
        # Suppress warnings temporarily
        warnings.filterwarnings("ignore")

        # Set initial paths
        self.results_path = "../results/results_{}.xlsx"
        self.image_base_path = "../results/regions/Images_{}/{}/High Quality/"

    def _load_sorted_data(self, region, tuned=False):
        """
        Load and sort the data for a given region based on accuracy.

        Parameters:
        -----------
        region : str
            Name of the region.
        tuned : bool, optional
            Flag to determine if results are for tuned models.
            Default is False.

        Returns:
        --------
        DataFrame
            Sorted dataframe based on "Accuracy 10% Pred".
        """
        tuned_path = "not_tuned" if not tuned else "tuned"
        data_path = self.results_path.format(tuned_path)
        df = pd.read_excel(data_path, sheet_name=region)
        return df.sort_values(by="Accuracy 10% Pred",
                              ascending=False).reset_index(drop=True)

    def _get_image_path(self, region, row, tuned=False):
        """
        Construct the path for the image based on region and model details.

        Parameters:
        -----------
        region : str
            Name of the region.
        row : pandas.Series
            Row from the dataframe containing model details.
        tuned : bool, optional
            Flag to determine if results are for tuned models.
            Default is False.

        Returns:
        --------
        str
            Path to the corresponding image.
        """
        tuned_path = "Not tuned" if not tuned else "Tuned"
        image_path = self.image_base_path.format(region, tuned_path)
        tuned_suffix = "1" if not tuned else "2_Tuned"
        file_name = (f"plot_4k_0{tuned_suffix}_{row['Model']}_"
                     f"Trained_on_{row['Trained on']}_{region}_new.png")
        return image_path + file_name

    def _apply_region_transform(self, region, img, tuned=False):
        """
        Apply transformations to the image based on the region.

        Parameters:
        -----------
        region : str
            Name of the region.
        img : ndarray
            Image data to be transformed.
        tuned : bool, optional
            Flag to determine if results are for tuned models.
            Default is False.

        Returns:
        --------
        ndarray
            Transformed image data.
        """
        if region == "UK":
            return img[
                int(0.0275 * img.shape[0]): -int(0.285 * img.shape[0]),
                int(0.20 * img.shape[1]): -int(0.22 * img.shape[1]), :]

        if region == "LONDON":
            y_scale, x_scale = 1.2, 0.8
            img = img[int(0.163 * img.shape[0]):, :, :]
        elif region == "WOODSTOCK":
            y_scale, x_scale = 1.3, 0.9
            img = img[int(0.134 * img.shape[0]):, :, :]
        elif region == "SOUTH SHIELDS":
            y_scale, x_scale = 1.4, 0.7
            img = img[int(0.245 * img.shape[0]):, :, :]

        return img

        return zoom(img, (y_scale, x_scale, 1))

    def _plot_images_for_region(self, region, df_sorted, axs, tuned=False):
        """
        Plot images for the specified region based on the sorted dataframe.

        Parameters:
        -----------
        region : str
            Name of the region to be plotted.
        df_sorted : DataFrame
            Dataframe containing sorted results for models.
        axs : array
            Array of axis objects for plotting.
        tuned : bool, optional
            Flag to determine if results are for tuned models.
            Default is False.
        """
        for i, row in df_sorted.iterrows():
            ax = axs[i % 2, i // 2] if region != "UK" else axs[i]
            img_path = self._get_image_path(region, row, tuned)
            img = imread(img_path)
            img = self._apply_region_transform(region, img, tuned)
            title_text = (f"{row['Model']} trained on {row['Trained on']}: "
                          f"{row['Accuracy 10% Pred']:.2f}% accuracy")
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title_text, fontsize=10)

    def plot_region_accuracy(self, region, tuned=False):
        """
        Plot accuracy images for the specified region.

        Parameters:
        -----------
        region : str
            Name of the region to be plotted.
        tuned : bool, optional
            Flag to determine if results are for tuned models.
            Default is False.
        """
        df_sorted = self._load_sorted_data(region, tuned)

        if region == "UK":
            fig, axs = plt.subplots(1, 5, figsize=(20, 10))
        else:
            fig, axs = plt.subplots(2, 5, figsize=(20, 10))
            order = [
                x for pair in zip(range(len(df_sorted) // 2),
                                  range(len(df_sorted) // 2,
                                        len(df_sorted)))
                for x in pair
            ]
            df_sorted = df_sorted.loc[order].reset_index(drop=True)

        self._plot_images_for_region(region, df_sorted, axs, tuned)
        plt.tight_layout()
        plt.show()


class DataFrameStyler:
    """
    Styles and displays dataframes visually.

    Attributes:
    -----------
    df : pandas.DataFrame
        Dataframe to style.
    gradient_colours : list, optional
        Gradient colours. Default ["#e06666", "#93c47d"].
    trained_on_colours : dict, optional
        colours for 'trained_on'. Default {'England': '#cfe2f3',
                                          'Others': '#fce5cd'}.

    Methods:
    --------
    display_styled_df(numerical_columns: list, display_output=True) -> str:
        Display or return styled dataframe.
    save_to_html(filename: str, numerical_columns: list) -> None:
        Save styled dataframe to HTML.
    """
    def __init__(self, df, gradient_colours=None, trained_on_colours=None):
        self.df = df.copy()
        self.gradient_colours = gradient_colours or ["#e06666", "#93c47d"]
        self.trained_on_colours = trained_on_colours or {
            'England': '#cfe2f3',
            'Others': '#fce5cd'   # Default colour
        }

    def _setup_dataframe(self):
        """
        Prepare the dataframe for styling by making necessary replacements
        and formatting adjustments.
        """
        self.df.replace('UK', 'England', inplace=True)
        self.df['Region'] = self.df['Region'].str.title()
        self.df['Trained on'] = self.df['Trained on'].str.title()
        if 'Tuned' in self.df.columns:
            self.df.drop(columns='Tuned', inplace=True)

    def _ensure_numeric_columns(self, cols):
        """
        Convert specified columns to numeric if they're detected as object
        type.

        Parameters:
        -----------
        cols : list
            List of column names to check and convert to numeric.
        """
        for col in cols:
            if self.df[col].dtype == 'object':
                self.df[col] = (self.df[col]
                                .str.replace('%', '')
                                .astype(float))

    def _highlight_trained_on(self, row):
        """
        Highlight rows based on the 'Trained on' value.

        Parameters:
        -----------
        row : pandas.Series
            Row from the dataframe being styled.

        Returns:
        --------
        list
            List of styles for each cell in the row.
        """
        styles = {col: '' for col in row.index}
        colour = (self.trained_on_colours[row['Trained on']]
                  if row['Trained on'] in self.trained_on_colours
                  else self.trained_on_colours['Others'])
        styles['Region'] = f'background-colour: {colour}'
        styles['Trained on'] = f'background-colour: {colour}'
        return [styles[col] for col in row.index]

    def _apply_styling(self, num_cols):
        """
        Apply styling to the dataframe.

        Parameters:
        -----------
        num_cols : list
            List of column names with numerical data to be styled.

        Returns:
        --------
        str
            Styled dataframe rendered as an HTML string.
        """
        align_center = {
            "selector": "th, td",
            "props": [("text-align", "center")]
        }

        cmap = LinearSegmentedColormap.from_list("custom",
                                                 self.gradient_colours)

        styled = (self.df.style.background_gradient(cmap=cmap, subset=num_cols)
                  .apply(self._highlight_trained_on, axis=1)
                  .set_table_styles([align_center])
                  .format("{:.2f}%", subset=num_cols)
                  .hide_index())  # Hiding the index

        return styled.render()

    def _generate_legend_row(self, background_colour, text):
        """
        Generate a single row of the legend table in HTML format.
        """
        return (
            f'<tr>'
            f'<td style="background-colour: {background_colour}; '
            f'width: 30px;"></td>'
            f'</tr>'
        )

    def display_styled_df(self, numerical_columns, display_output=True):
        """
        Display or return the styled dataframe with the given numerical
        columns.

        Parameters:
        -----------
        numerical_columns : list
            List of column names with numerical data to be styled.
        display_output : bool, optional
            If True, the method will display the styled dataframe. If False,
            it will return the styled dataframe as an HTML string. Default is
            True.

        Returns:
        --------
        str, optional
            If display_output is False, returns the styled dataframe as an
            HTML string.
        """
        self._setup_dataframe()
        self._ensure_numeric_columns(numerical_columns)

        styled_html = self._apply_styling(numerical_columns)

        # Using the private method to generate the legend table
        legend_rows = [
            (self.trained_on_colours['England'], 'Trained on England'),
            (self.trained_on_colours['Others'], 'Trained on Others'),
            ('', '<br>'),  # Separator row
            (self.gradient_colours[0], 'Lower Value'),
            (self.gradient_colours[1], 'Higher Value')
        ]

        legend_html = '<table style="border:0px; margin-left:20px;">'
        for colour, text in legend_rows:
            legend_html += self._generate_legend_row(colour, text)
        legend_html += '</table>'

        combined_html = (
            f'<div style="display:flex; direction:row;">'
            f'{styled_html}{legend_html}</div>'
        )

        if display_output:
            display(HTML(combined_html))
        else:
            return combined_html

    def save_to_html(self, filename, numerical_columns):
        """
        Save the styled dataframe to an HTML file.

        Parameters:
        -----------
        filename : str
            Name of the output HTML file.
        numerical_columns : list
            List of column names with numerical data to be styled.

        Returns:
        --------
        None
        """
        combined_html = self.display_styled_df(numerical_columns,
                                               display_output=False)

        # Define column widths
        col_widths = {
            "Region": "10%",
            "DT": "12%",
            "LR": "12%",
            "XGBoost": "12%",
            "LightGBM": "12%",
            "RNN": "12%",
            "Trained on": "10%"
        }

        for col, width in col_widths.items():
            old_str = (f'<th class="col_heading level0 col0">{col}</th>')
            new_str = (f'<th style="width:{width}" '
                       f'class="col_heading level0 col0">{col}</th>')
            combined_html = combined_html.replace(old_str, new_str)

        with open(filename, 'w', encoding="utf-8") as f:
            f.write(combined_html)


class PropertyPricePlotManager:

    def __init__(self, model, postcode_mapping, house_data, geo_data, cities,
                 percentage=10):
        self.model = model
        self.postcode_mapping = postcode_mapping
        self.house_data = house_data
        self.geo_data = geo_data
        self.cities = cities
        self.percentage = percentage

    @staticmethod
    def _add_postcode_prefix(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a new column 'Postcode_prefix' to the DataFrame based on the
        Postcode values.

        Parameters:
        - df: Input DataFrame containing 'Postcode' column.

        Returns:
        - DataFrame with added 'Postcode_prefix' column.
        """
        df['Postcode_prefix'] = df['Postcode'].apply(
            lambda x: x[0] if x[1].isdigit() else x[:2]
        )
        return df

    @staticmethod
    def _get_city_data(colourize_data: ColourizePredictionsDataset,
                       house_data: pd.DataFrame,
                       city: str) -> (pd.DataFrame, pd.DataFrame):
        """
        Retrieve colour and relevance subsets of data for a given city.

        Parameters:
        - colourize_data: Dataset with colourization information.
        - house_data: Dataset with house information.
        - city: Name of the city for which data subsets are retrieved.

        Returns:
        - Tuple of two DataFrames: city_colours_subset and
        city_relevance_subset.
        """
        colourize_dataset = PropertyPricePlotManager._add_postcode_prefix(
            colourize_data._merged.copy()
        )
        relevance_dataset = PropertyPricePlotManager._add_postcode_prefix(
            colourize_data._df.copy()
        )

        if city == "LONDON":
            london_prefix = ['E', 'EC', 'N', 'NW', 'SE', 'SW', 'W', 'WC']
            city_colours_subset = colourize_dataset[
                colourize_dataset["Postcode_prefix"].isin(london_prefix)
            ]
            city_relevance_subset = relevance_dataset[
                relevance_dataset["Postcode_prefix"].isin(london_prefix)
            ]
        else:
            minor_postcodes = house_data[
                house_data['Town_City'] == city
            ]['Postcode'].unique().tolist()

            minor_postcodes = [x[:-2] for x in minor_postcodes]
            city_colours_subset = colourize_dataset[
                colourize_dataset["Postcode"].str.startswith(tuple(
                    minor_postcodes
                ))
            ]
            city_relevance_subset = relevance_dataset[
                relevance_dataset["Postcode"].str.startswith(tuple(
                    minor_postcodes
                ))
            ]

        return city_colours_subset, city_relevance_subset

    def _plot_for_uk(self,
                     colourize_data: ColourizePredictionsDataset,
                     tuned: bool) -> None:
        """
        Plot the property price map for the entire UK.

        Parameters:
        - colourize_data (ColourizePredictionsDataset): The dataset to use for
        plotting.
        - tuned (bool): If the model has been tuned. Default is False.

        Returns:
        None
        """
        postcodes_counts = colourize_data.postcode_counts.sort_values(
            by=['Efficiency', 'Total'], ascending=[False, False]
        )
        colourize_data._count_relevant()
        plotter = PropertyPriceMapPlotter(
            colourize_data._merged, sizes=(20, 12),
            model=str(self.model),
            accuracy=colourize_data.accuracy, tuned=tuned
        )
        plotter.plot(percentage=10, plot_bars=True)

    def _plot_for_city(self,
                       colourize_data: ColourizePredictionsDataset,
                       city: str,
                       tuned: bool,
                       trained_on: str = "UK") -> None:
        """
        Plot the property price map for a specified city.

        Parameters:
        - colourize_data (ColourizePredictionsDataset): The dataset to use for
        plotting.
        - city (str): The name of the city to plot.
        - tuned (bool): If the model has been tuned. Default is False.
        - trained_on (str): The dataset the model was trained on. Default is
        "UK".

        Returns:
        None
        """
        colourize_data_temp = copy.deepcopy(colourize_data)

        dataframes = PropertyPricePlotManager._get_city_data(
            colourize_data, self.house_data, city
        )

        colourize_data_temp._merged, colourize_data_temp._df = dataframes

        colourize_data_temp._process_postcode_efficiency()
        colourize_data_temp._count_relevant()

        plotter = PropertyPriceMapPlotter(
            colourize_data_temp._merged, area=city,
            model=str(self.model),
            accuracy=colourize_data_temp.accuracy, sizes=(20, 20),
            tuned=tuned, trained_on=trained_on
        )
        plotter.plot(percentage=10, plot_edges=True, plot_text=False)

    def get_results(self, tuned: bool = False,) -> None:
        """
        Get the results for the specified cities based on the predictions and
        the original test data, and plot them.

        Parameters:
        - tuned (bool): If the model has been tuned. Default is False.

        Returns:
        None
        """
        colourize_data = ColourizePredictionsDataset(
            self.model.original_y_test,
            self.model.predictions_test,
            self.postcode_mapping,
            self.house_data,
            self.geo_data,
            percentage=self.percentage
        )

        if len(self.cities) > 1:
            self._plot_for_uk(colourize_data, tuned)
            for city in self.cities:
                self._plot_for_city(colourize_data, city, tuned)
        else:
            self._plot_for_city(
                colourize_data,
                self.cities[0],
                tuned,
                trained_on=self.cities[0]
            )
