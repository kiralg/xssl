# Functions for evaluating CNR over range data

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt
import os
import gc
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm


def read_data(filename: str) -> pd.DataFrame:
    """
        Read Sea Surface Lidar (SSL) measurement data from a CSV file.

        This function imports SSL scan results stored in a semicolon-separated CSV file and returns a cleaned
        pandas DataFrame. The azimuth values are rounded and normalised to the range [0, 360) degrees.

        The dataset is expected to include measurements for multiple timestamps and contains at least the following
        columns:
            - 'timestamp' (str or datetime-like): Unique time identifier for each scan.
            - 'azimuth' (float): Azimuth angle, rounded to nearest integer [deg].
            - 'elevation' (float): Elevation angle in degrees.
            - 'range' (float): Distance from the lidar [m].
            - 'cnr' (float): Carrier-to-noise ratio [dB].

        Args:
            filename (str): Path to the CSV file containing SSL measurement data.

        Returns:
            pandas.DataFrame: Cleaned DataFrame containing all original columns, with 'azimuth' rounded and normalised.
        """
    data= pd.read_csv(filename, delimiter=';', low_memory=False)
    data = data.round({'azimuth': 0})
    data['azimuth'] = pd.to_numeric(data['azimuth'], errors='coerce')
    data['azimuth'] = data['azimuth'] % 360
    return data


# Function to describe the cnr over range values
def sigmoid_lin_function(x: np.ndarray, up: float, down: float, mid: float, growth: float, a: float) -> np.ndarray:
    """
     Computes a modified inverse sigmoid function with an additional linear term for detecting
     the inflection point.

    This function is used to model Carrier to noise (CNR) over range signal of the laser beam. It combines a standard
    sigmoid curve with a linear scaling factor (1 + a * x) to detect the inflection point corresponding to the sea surface.
    To avoid numerical overflow, the exponent of the exponential function is clipped to a finite range.

    Args:
        x (array-like): Range values [m].
        up (float): Upper asymptote of the sigmoid.
        down (float): Lower asymptote of the sigmoid.
        mid (float): Inflection point of the sigmoid [m].
        growth (float): Growth rate controlling the steepness of the sigmoid.
        a (float): Linear scaling coefficient.

    Returns:
        numpy.ndarray: Modelled CNR values corresponding to the input range.
    """
    exponent = np.clip((x - mid) * growth, -500, 500)  # Limit exponent to prevent overflow
    return (up - down) * (1 + a * x) / (1 + np.exp(exponent)) + down


# Function to find the distance to the sea surface
def find_distance_to_surface(data: pd.DataFrame, bound_data: pd.DataFrame, func=sigmoid_lin_function,
                             initial_guess=[0.1, 0.1]) -> tuple[np.ndarray, float, float]:
    """
    Estimate the distance to the sea surface by fitting a CNR-over-range model.

    This function fits a sigmoid-based model to CNR-over-range data in order to
    determine the inflection point corresponding to the sea surface. Parameter
    bounds are selected based on the elevation angle of the scan.

    Args:
        data (pandas.DataFrame): CNR-over-range data for a single scan. Required columns:
            - 'range'
            - 'cnr'
            - 'elevation'
        bound_data (pandas.DataFrame): Table defining lower and upper bounds for the
            model fit parameters as a function of elevation angle. The DataFrame
            must contain:

            - 'Elev' (float): Elevation angle corresponding to the bounds.
            - One lower-bound column per model parameter, named '<param>_lb'.
            - One upper-bound column per model parameter, named '<param>_ub'.

            The number and order of bound columns must match the parameter order
            expected by the model function `func`. For example, if `func` expects
            parameters [p0, p1, p2, p3, p4], then `bound_data` must contain:

                - p0_lb, p1_lb, p2_lb, p3_lb, p4_lb
                - p0_ub, p1_ub, p2_ub, p3_ub, p4_ub

            For a given scan, the row with the closest 'Elev' value to the scan's
            elevation angle is selected, and its bounds are used for the optimisation.
        func (callable, optional): Model function used for fitting.
            Defaults to `sigmoid_lin_function`.
        initial_guess (list, optional): Initial guess values for the growth-related
            parameters of the model.

    Returns:
        tuple: A tuple containing:
            - res.x (numpy.ndarray): Optimised model parameters.
            - high_cnr (float): Maximum CNR value in the scan.
            - first_cnr (float): First CNR value of the scan.
    """

    timestamp_data = data
    fit_function = lambda *params: func(timestamp_data['range'], *params)
    cost_function = lambda params: np.sum((timestamp_data['cnr'] - fit_function(*params)) ** 2)

    # Bound depending on chosen elevation angle
    elev = timestamp_data['elevation'].iloc[0]
    idx = (bound_data['Elev'] - elev).abs().idxmin()
    bound_row = bound_data.loc[idx]

    lower_bounds = [bound_row[col] for col in bound_data.columns if col.endswith('_lb')]
    upper_bounds = [bound_row[col] for col in bound_data.columns if col.endswith('_ub')]
    bounds = Bounds(lower_bounds, upper_bounds)

    # First CNR Value for Filter
    first_cnr = timestamp_data['cnr'].iloc[0]
    high_cnr = timestamp_data['cnr'].max()
    low_cnr = timestamp_data['cnr'].min()
    middle_cnr = (high_cnr + low_cnr) / 2
    min_distance = timestamp_data['range'][timestamp_data['cnr'] > middle_cnr].min()
    middle_range = timestamp_data['range'][
        (timestamp_data['range'] > min_distance) & (timestamp_data['cnr'] <= middle_cnr)].min()

    initial_guess = [high_cnr, low_cnr, middle_range] + initial_guess
    res = minimize(cost_function, initial_guess, bounds=bounds)
    return res.x, high_cnr, first_cnr


# Function for plotting CNR over range data
def plot_cnr_over_range(data:pd.DataFrame, res_fit_function, folder:str, timestamp_str: str,
                        fit_function=sigmoid_lin_function):
    """
    Plot CNR as a function of range together with the fitted model.

    This function visualises the CNR-over-range data for a single scan and overlays the fitted sigmoid-based model.
    The estimated surface distance (inflection point of the fitted curve) is highlighted, and the resulting plot
    is saved to disk.

    Args:
        data (pandas.DataFrame): CNR-over-range data for a single scan.
            The DataFrame must contain the following columns:
            - 'range' (float): Distance along the beam.
            - 'cnr' (float): Carrier-to-noise ratio values.
            - 'elevation' (float): Elevation angle of the scan (assumed constant within the DataFrame).
            - 'azimuth' (float): Azimuth angle of the scan (assumed constant within the DataFrame).

        res_fit_function (array-like): Optimised model parameters returned by the
            fitting routine. The parameter order must match the signature of `fit_function`.

        folder (str): Output directory where the plot image is saved.

        timestamp_str (str): Timestamp of the scan, used for plot labelling. Must be convertible to a pandas datetime.

        fit_function (callable, optional): Model function used for plotting. The function must accept the range array
            as first argument followed by the model parameters. Defaults to `sigmoid_lin_function`.

    """
    # Convert timestamp string to a readable format for the plot title
    new_time_str = pd.to_datetime(timestamp_str).strftime("%d-%m-%y %H:%M:%S")

    # Alias for clarity
    timestamp_data = data
    res = res_fit_function

    # Extract scan metadata (constant per scan)
    elevation_value = timestamp_data['elevation'].iloc[0]
    azimuth_value = round(timestamp_data['azimuth'].iloc[0])  # Round the azimuth value

    # Create figure
    plt.figure()

    # Plot CNR over range data
    plt.plot(timestamp_data['range'], timestamp_data['cnr'], '.', linewidth=8, markersize=8,
             color='darkblue')

    # Plot fitted sigmoid-function
    range_range = np.linspace(timestamp_data['range'].min(), timestamp_data['range'].max(), 1000)
    fitted_cnr = fit_function(range_range, res[0], res[1], res[2], res[3], res[4])
    plt.plot(range_range, fitted_cnr, '-', linewidth=3, color='k')  # Black line

    # Plot function's inflection point
    plt.vlines(res[2], timestamp_data['cnr'].min() - 1, timestamp_data['cnr'].max() + 1, color='r', linewidth=3)
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel('Range [m]', fontsize=16)
    plt.ylabel('CNR [dB]', fontsize=16)
    plt.title(fr'Date: {new_time_str}, $\varphi$ = {round(elevation_value,2)}, $\Theta$ = {azimuth_value}', fontsize=16)

    # Plot xlim
    plt.xlim(timestamp_data['range'].min(), res[2]+500)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()

    # Create a folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save plot with scan-specific filename
    plot_file_name = os.path.join(folder, f'CNR_over_Range_Azim_{azimuth_value}_Elev_{round(elevation_value, 2)}.png')
    plt.savefig(plot_file_name)

    # Clean up to avoid memory leaks in batch processing
    plt.clf()
    plt.close('all')
    gc.collect()


# Function for analysing CNR over range data
def analyse_cnr_over_range(filtered_data: pd.DataFrame, bounds: pd.DataFrame, folder_out:str,
                           min_cnr:float=-30, show_plot:bool=True) -> pd.DataFrame:
    """
    Analyse CNR-over-range data to estimate the distance to the sea surface.

    This function applies quality filtering, fits a CNR-over-range model to a single scan, optionally generates
    plots of cnr over range data, and returns the derived sea surface distance together with additional fit parameters and
    quality metrics.

    Args:
        filtered_data (pandas.DataFrame): Filtered CNR-over-range data for a single scan.
            Required columns:
            - 'range' (float): Distance along the beam
            - 'cnr' (float): Carrier to noise ratio signal
            - 'elevation' (float): Elevation angle (constant within scan)
            - 'azimuth' (float): Azimuth angle (constant within scan)
            - 'timestamp' (datetime): Timestamp of the scan
        bounds (pandas.DataFrame): Parameter bounds for the model fit.
        folder_out (str): Output directory for saving plots.
        min_cnr (float, optional): Minimum required CNR value for processing.
            Defaults to -30.
        show_plot (bool, optional): Whether to generate and save cnr over range plots with the estimated distance.
            Defaults to True.

    Returns:
        pandas.DataFrame: Single-row DataFrame containing:
        - 'Timestamp': Scan timestamp (string formatted)
        - 'Elevation': Elevation angle
        - 'Azimuth': Azimuth angle
        - 'Distance': Estimated sea surface distance
        - 'Growth': Growth parameter of the fit
        - 'CNR_max': Maximum CNR value
        - 'CNR_first': First CNR value
        - 'A', 'up', 'down', 'mid': Fitted model parameters

    Notes:
        - If the maximum CNR is below `min_cnr`, no fit is performed and NaN values
          are returned for the fit parameters.
        - The function assumes that `filtered_data` contains exactly one scan.
    """
    results_data = []

    # Extract scan metadata (assumed constant per scan)
    elevation_value = filtered_data['elevation'].iloc[0]
    azimuth_value = round(filtered_data['azimuth'].iloc[0])
    timestamp = filtered_data['timestamp'].iloc[0]
    timestamp_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')

    # Quality check: skip fitting if the maximum CNR of the scan is below the threshold
    if filtered_data['cnr'].max() < min_cnr:
        distance = np.nan
        cnr_max = np.nanmax(filtered_data['cnr'])
        cnr_first = np.nan
        # Placeholder for missing fit parameters
        res = [np.nan, np.nan, np.nan, np.nan, np.nan]

    else:
        # Fit CNR-over-range model to estimate surface distance
        res, cnr_max, cnr_first = find_distance_to_surface(data=filtered_data, bound_data=bounds)
        # Reject unphysical fits based on CNR threshold
        distance = np.nan if res[0] < min_cnr - 3 else res[2]

        # Optional visualisation of the fitted results
        if show_plot:
            plot_cnr_over_range(data=filtered_data, res_fit_function=res,
                            folder=folder_out, timestamp_str=timestamp_str)

    # Collect results in a single-row DataFrame
    results_data.append({'Timestamp': timestamp_str,
                         'Elevation': elevation_value,
                         'Azimuth': azimuth_value,
                         'Distance': distance,
                         'Growth': res[3],
                         'CNR_max': cnr_max,
                         'CNR_first': cnr_first,
                         'A': res[4],
                         'up': res[0],
                         'down': res[1],
                         'mid': res[2]})

    return pd.DataFrame(results_data)


# Function for parallel application of the cnr over range data analysis
def wrapper_parallel_distance_analysis(data_cnr:pd.DataFrame, data_bound:pd.DataFrame, file_out:str,
    cnr_threshold:float=-30, show_plot:bool=True, num_cpu:int =6) -> pd.DataFrame:
    """
        Perform parallel CNR-over-range analysis (function: 'analyse_cnr_over_range') for multiple timestamps.

        This function splits the input CNR dataset by timestamp and processes each scan independently using
        multiprocessing. For each timestamp, a CNR-over-range model is fitted to estimate the distance to the
        sea surface. The results are aggregated and written to a CSV file.

        Args:
            data_cnr (pandas.DataFrame): CNR-over-range data for multiple scans.
                The DataFrame must contain the following columns:
                - 'range' (float): Distance along the beam.
                - 'cnr' (float): Carrier-to-noise ratio.
                - 'elevation' (float): Elevation angle of the scan
                  (constant per timestamp).
                - 'azimuth' (float): Azimuth angle of the scan
                  (constant per timestamp).
                - 'timestamp' (datetime-like): Timestamp identifying each scan.
            data_bound (pandas.DataFrame): Parameter bounds for the model fit.
            file_out (str): Output directory for results and plots.
            cnr_threshold (float, optional): Minimum required maximum CNR value for a scan to be processed.
                Defaults to -30.
            show_plot (bool, optional): If True, generate and save diagnostic CNR-over-range plots for each
                processed scan. Defaults to True.
            num_cpu (int, optional): Number of parallel worker processes to use. Should not exceed the number of
                available CPU cores. Defaults to 6.

        Returns:
            pandas.DataFrame: Combined results of the CNR-over-range analysis for all processed timestamps.
        """

    # Extract unique timestamps to define independent scans
    unique_timestamps = data_cnr['timestamp'].unique()
    print(f'Processing timestamps: {unique_timestamps}')

    # Number of parallel worker processes
    num_workers = num_cpu

    # Create an output directory for CNR-over-range plots
    plots_dir = os.path.join(file_out, "plots_cnr_over_range")
    os.makedirs(plots_dir, exist_ok=True)

    # Prepare a multiprocessing pool
    with Pool(processes=num_workers) as pool, Manager() as manager:
        # Create one task per timestamp (one scan per task)
        tasks = [(data_cnr[data_cnr['timestamp'] == ts], data_bound,
                  plots_dir, cnr_threshold, show_plot)
                 for ts in unique_timestamps]

        results = []
        pbar = tqdm(total=len(tasks), desc="Processing timestamps")

        def collect_result(result):
            """Callback function to collect results from worker processes."""
            results.append(result)
            pbar.update()

        # Launch all tasks asynchronously
        for task in tasks:
            pool.apply_async(analyse_cnr_over_range, task, callback=collect_result)

        pool.close() # Stop accepting new tasks
        pool.join() # Wait for all worker processes to finish
        pbar.close() # Close the progress bar

    # Combine results from all timestamps into a single DataFrame
    final_results = pd.concat(results, ignore_index=True)

    # Save aggregated results to CSV
    output_file = os.path.join(plots_dir, 'results_cnr_over_range.csv')
    final_results.to_csv(output_file, index=False, sep=';')

    return final_results
