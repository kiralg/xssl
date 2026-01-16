# Function for the extended Sea Surface Levelling Method

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ======================================================================================================================
# Internal helper functions
# ======================================================================================================================

# Function to read SSL data from a CSV file
def read_data_ssl(filename: str) -> pd.DataFrame:
    """
    Read Sea Surface Lidar (SSL) measurement data from a CSV file.

    This function imports SSL scan results stored in a semicolon-separated CSV file and returns the contents as a
    pandas DataFrame. The file is expected to contain time-resolved lidar measurements including angular information
    and derived signal characteristics.

    Args:
        filename (str): Path to the CSV file containing the SSL measurement data.

    Returns:
        pandas.DataFrame: DataFrame containing the SSL scan results. Expected columns include:
            - 'Timestamp'
            - 'Azimuth'
            - 'Elevation'
            - 'Distance'
            - 'Growth'
            - 'CNR_max'
    """
    ssl_scan_results = pd.read_csv(filename, delimiter=';')
    return ssl_scan_results

# Function to extract the first and last timestamp from an SSL dataset
def get_times(ssl_scan_results: pd.DataFrame) -> tuple:
    """
    Extract the first and last timestamp from an SSL dataset.

    This function converts the 'Timestamp' column of the input DataFrame to pandas datetime objects and returns
    the earliest and latest timestamps present in the dataset.

    Args:
        ssl_scan_results (pandas.DataFrame): SSL measurement data containing
            a 'Timestamp' column.

    Returns:
        tuple: Tuple containing:
            - first_timestamp (pandas.Timestamp): Timestamp of the first measurement.
            - last_timestamp (pandas.Timestamp): Timestamp of the last measurement.
    """
    times = pd.to_datetime(ssl_scan_results['Timestamp'])
    first_timestamp = times.iloc[0]
    last_timestamp = times.iloc[-1]
    return first_timestamp, last_timestamp

# Function to filter SSL data based on the maximum Carrier-to-Noise Ratio (CNR)
def cnr_max_filter(ssl_scan_results: pd.DataFrame, cnr_max: float) -> pd.DataFrame:
    """
        Filter SSL data based on the maximum Carrier-to-Noise Ratio (CNR).

        This function removes all measurements for which the maximum CNR value exceeds a specified threshold. It is
        typically used to exclude measurements affected by excessive noise or non-physical signal behavior.

        Args:
            ssl_scan_results (pandas.DataFrame): SSL measurement data. Required columns:
                - 'Timestamp'
                - 'Elevation'
                - 'Azimuth'
                - 'Distance'
                - 'Growth'
                - 'CNR_max'
            cnr_max (float): Upper threshold for the maximum CNR value.

        Returns:
            pandas.DataFrame: Filtered SSL dataset containing only measurements with
            'CNR_max' values smaller than the specified threshold.
    """
    ssl_scan_filtered = ssl_scan_results.loc[ssl_scan_results['CNR_max'] < cnr_max]
    return ssl_scan_filtered

# Function to filter SSL data based on the growth parameter
def growth_filter(ssl_scan_results: pd.DataFrame, growth_min: float, growth_max: float) -> pd.DataFrame:
    """
        Filter SSL data based on the growth parameter.

        This function retains only those measurements whose 'Growth' values lie within a specified interval.
        The growth parameter typically characterizes the signal behavior of the CNR over range and is used to exclude
        unreliable measurements.

        Args:
            ssl_scan_results (pandas.DataFrame): SSL measurement data. Required columns:
                - 'Timestamp'
                - 'Elevation'
                - 'Azimuth'
                - 'Distance'
                - 'Growth'
                - 'CNR_max'
            growth_min (float): Lower threshold for the growth parameter.
            growth_max (float): Upper threshold for the growth parameter.

        Returns:
            pandas.DataFrame: Filtered SSL dataset containing only measurements with 'Growth' values between
            growth_min and growth_max.
    """
    ssl_scan_filtered = ssl_scan_results.loc[ssl_scan_results['Growth'] < growth_max]
    ssl_scan_filtered = ssl_scan_filtered.loc[ssl_scan_filtered['Growth'] > growth_min]
    return ssl_scan_filtered

# ======================================================================================================================
# Functions
# ======================================================================================================================

# Extended Sea Surface Levelling (SSL) function
def extended_ssl_fun(azi_dist: list, pitch: float, roll: float, h_lidar: float, elev_offset: float):
    '''

    Compute the internal lidar elevation angle required to target a fixed external point using the extended Sea Surface
    Levelling (SSL) method under small-angle approximations.

    This function represents a derived formulation of the extended Sea Surface Levelling method. Assuming small pitch,
    roll and elevation angles, it establishes a relationship between the internal lidar elevation angle and the
    internal lidar azimuth angle, while accounting for instrument tilt (pitch and roll), lidar height, a constant
    elevation offset and the curvature of the Earth.

    Pitch and roll are defined in a left-oriented coordinate system:
        - Pitch is positive when the instrument is tilted downward towards the North, corresponding to a rotation
          about the South–West axis.
        - Roll is positive when the instrument is tilted downward towards the West, corresponding to a rotation about
          the North–South axis.

    The elevation offset represents a systematic vertical displacement of the laser beam
    caused by internal misalignments:
        - A positive offset means that the laser beam is shifted upward in the instrument coordinate system.
        - Consequently, a positive offset requires a negative correction of the internally set elevation angle
          in order to reach the same external target.

    Args:
        azi_dist (list or tuple): Two-element sequence containing:
            - azimuth (numpy.ndarray or float): Internal azimuth angle(s) [deg].
            - distance (numpy.ndarray or float): Distance(s) to the sea surface [m].
        pitch (float): Pitch angle of the lidar system [deg].
        roll (float): Roll angle of the lidar system [deg].
        h_lidar (float): Height of the lidar above the sea surface [m].
        elev_offset (float): Constant elevation offset due to internal misalignment [deg].

    Returns:
        numpy.ndarray: Internal elevation angle [deg] required to intersect the sea surface at the specified distance
        and azimuth, accounting for pitch, roll, lidar height, Earth curvature, and elevation offset.

    Notes:
        - The formulation assumes small pitch, roll, and elevation angles (small-angle approximation).
        - Earth curvature is approximated by d² / (2R), with R = 6,371,000 m.


    '''

    azimuth, distance = azi_dist
    # earth curvature depending on the distance
    earthcurv = distance**2/(2* 6371000)

    return pitch * np.cos(azimuth * np.pi/180) - roll * np.sin(azimuth * np.pi/180) - \
        ((h_lidar-earthcurv)/distance) / np.pi * 180 - elev_offset

# Function to optimize the data to determine pitch, roll, lidar height and elevation offset
def curve_fit_offset(ssl_scan_results: pd.DataFrame, pitch_roll_offset_fun, initial_guess=None) -> tuple:
    """
    Optimize SSL data to determine pitch, roll, lidar height, and elevation offset.

    This function uses `scipy.optimize.curve_fit` to estimate the optimal parameters describing the relationship between
    pitch, roll, lidar height, and a constant elevation offset on the measured elevation angles. All available azimuth,
    elevation, and distance data from the SSL scan are used in the fit.
    For the `pitch_roll_offset_fun` function, the `extended_ssl_fun` function defined above is used, here.

    Args:
        ssl_scan_results (pandas.DataFrame): SSL measurement data containing at least the following columns:
            - 'Timestamp'
            - 'Azimuth'
            - 'Elevation'
            - 'Distance'
            - 'Growth'
        pitch_roll_offset_fun (callable): Function that models the dependence of the elevation angle
            on azimuth, distance, pitch, roll, lidar height, and elevation offset.
            Signature must be `f([azimuth, distance], pitch, roll, h_lidar, elev_offset)`.
        initial_guess (list, optional): Initial guess values for the optimization
            `[pitch, roll, lidar_height, elevation_offset]`. Defaults to `[0.1, 0.1, 20, -0.2]`.

    Returns:
        tuple: A tuple containing:
            - params (numpy.ndarray): Optimized parameter values `[pitch, roll, lidar_height, elevation_offset]`.
            - params_covariance (numpy.ndarray): Covariance matrix of the optimized parameters,
              which can be used to estimate parameter uncertainties.

    """
    if initial_guess is None:
        initial_guess = [0.1, 0.1, 20, -0.2]
    azim_angle = ssl_scan_results['Azimuth'].values
    # distance values in data are negative
    dist = ssl_scan_results['Distance'].values
    elev_angle = ssl_scan_results['Elevation'].values
    # noinspection PyTupleAssignmentBalance
    params, params_covariance = optimize.curve_fit(pitch_roll_offset_fun,
                                                   [azim_angle, dist],
                                                   elev_angle,
                                                   p0=initial_guess,
                                                   maxfev=10000)
    return params, params_covariance

# Sinus and cosinus function for tilt correction
def sin_cos_fun(azimuth: np.array, pitch: float, roll: float, offset: float):
    '''
    Compute the internal elevation angle adjustment required to target a fixed external direction as a function of
    azimuth, pitch, roll, and an elevation offset.

    This function models the effect of instrument tilt on the measured elevation angle. The correction is expressed as
    a sinusoidal function of azimuth, where pitch and roll contribute via cosine and sine terms, respectively.
    An additional constant offset accounts for a systematic elevation bias.

    Pitch and roll are defined in a left-oriented coordinate system:
        - Pitch is positive when the instrument is tilted downward towards the North, corresponding to a rotation
          about the South–West axis.
        - Roll is positive when the instrument is tilted downward towards the West, corresponding to a rotation about
          the North–South axis.

    The elevation offset represents a systematic vertical displacement of the laser beam
    caused by internal misalignments:
        - A positive offset means that the laser beam is shifted upward in the instrument coordinate system.
        - Consequently, a positive offset requires a negative correction of the internally set elevation angle
          in order to reach the same external target.

    Args:
        azimuth (numpy.ndarray or array-like): Azimuth angle(s) [deg].
        pitch (float): Pitch angle of the instrument [deg].
        roll (float): Roll angle of the instrument [deg].
        offset (float): Constant elevation offset due to internal vertical beam displacement [deg].

    Returns:
        numpy.ndarray: Elevation angle adjustment [deg] as a function of azimuth. These values describe the internal
        elevation angles (or elevation corrections) that must be set to correctly point the laser beam at a fixed
        external location in the earth-fixed coordinate system.

    '''
    return pitch * np.cos(azimuth * np.pi/180) - roll * np.sin(azimuth * np.pi/180) - offset


# Function to calculate untilted elevation angles values
def ssl_results_elev_untilted(ssl_results: pd.DataFrame, pitch: float, roll: float,
                              elevation_offset: float) -> pd.DataFrame:
    """
    Compute tilt-corrected ("untilted") elevation angles for SSL measurements.

    This function removes the effect of instrument tilt (pitch and roll) and a constant elevation offset from the
    measured elevation angles. The tilt contribution is calculated as a function of azimuth, pitch, roll, and
    elevation offset using `sin_cos_fun`.
    The resulting tilt is then subtracted from the measured elevation to obtain the untilted elevation angle.

    Args:
        ssl_results (pandas.DataFrame): SSL measurement data. Required columns:
            - 'Azimuth' (float): Azimuth angle [deg].
            - 'Elevation' (float): Measured elevation angle [deg].
        pitch (float): Pitch angle of the lidar system [deg].
        roll (float): Roll angle of the lidar system [deg].
        elevation_offset (float): Constant offset in elevation angle [deg].


    Returns:
        pandas.DataFrame: The input DataFrame with additional columns:
            - 'Tilt' (float): Computed tilt contribution [deg].
            - 'Elevation Untilted' (float): Tilt-corrected elevation angle [deg].

        Notes:
        - Tilt correction removes the combined effect of pitch, roll, and elevation offset from the measured
          elevation angle.

    """
    ssl_results = ssl_results.copy()
    ssl_results['Tilt'] = sin_cos_fun(azimuth=ssl_results['Azimuth'],
                                      pitch=pitch,
                                      roll=roll,
                                      offset=elevation_offset)
    ssl_results['Elevation Untilted'] = ssl_results['Elevation'] - ssl_results['Tilt']
    return ssl_results


# Function to calculate the elevation angle depending on distance values and a known lidar height
def elev_depending_dist(dist_var: np.array, h_lidar: float) -> np.array:
    """
    Calculate the elevation angle as a function of distance and lidar height.

    The elevation angle is computed assuming a spherical Earth and accounts for Earth curvature.
    Positive distances are measured from the scanning lidar to the water surface.

    Args:
        dist_var (np.array): Horizontal distance from the lidar to the water surface in meters. Values must be positive.
        h_lidar (float): Height of the lidar above the sea surface in meters.

    Returns:
        Elevation angle in degrees. Negative values indicate downward-looking angles.

    """
    earthcurv = dist_var ** 2 / (2 * 6371000)
    delta_elev = np.arcsin(-(h_lidar-earthcurv)/dist_var) / np.pi * 180
    return delta_elev


# Function to determine pitch, roll and elevation angles
def ssl_wrapper(ssl_data: pd.DataFrame, distance: list=None, elevation: list=None, azimuth: list=None,
                growth:list=None, cnr_max: list=None, distance_correct: float= None,
                azimuth_exclude: list=None) -> tuple:
    """
    Process Sea Surface Lidar (SSL) data to determine the device orientation (pitch and roll) and the elevation offset
    using the 'extended_ssl_fun' function.
    The SSL dataset consists of distances to the sea surface derived from the carrier-to-noise ratio (CNR) over
    range signal, together with the corresponding elevation and azimuth angles. It is important that measurements of
    several elevations and azimuth angles are available.

    This function filters the input SSL data based on distance, elevation, azimuth, CNR, and growth parameters,
    optionally applies a distance correction and then uses a curve fitting approach to estimate the pitch, roll,
    lidar height and elevation offset. It also computes tilt-corrected ("untilted") elevation angles and the
    root mean square error (RMSE) of the elevation fit.

    Args:
        ssl_data (pandas.DataFrame): SSL measurement data. Required columns:
            - 'Timestamp' (str): Measurement timestamp.
            - 'Distance' (float): Distance to the sea surface [m].
            - 'Elevation' (float): Measured elevation angle [deg].
            - 'Azimuth' (float): Azimuth angle [deg].
            - 'CNR_max' (float): Maximum Carrier-to-Noise Ratio.
            - 'Growth' (float): Growth parameter from the CNR over range analysis.
        distance (list, optional): [min, max] distances to consider [m]. Defaults to None.
        elevation (list, optional): [min, max] elevation angles to consider [deg]. Defaults to None.
        azimuth (list, optional): [min, max] azimuth angles to consider [deg]. Defaults to None.
        growth (list, optional): [min, max] growth values to consider. Defaults to None.
        cnr_max (list, optional): [min, max] the maximal CNR value in the cnr over range curve should be be greater
        than cnr_max[0] and smaller than cnr_max[1].
        distance_correct: Variable that corrects the distance
        azimuth_exclude (list or float, optional): Azimuth angles to exclude from analysis. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - result_data (dict): Optimized parameters and metadata:
                - 'Pitch', 'Roll', 'Lidar Height', 'Elevation Offset'
                - 'Number of Values', 'RMSE', uncertainties for each parameter
                - 'Timestamp' and 'last Timestamp'
                - 'Distance Correction'
            - ssl_data (pandas.DataFrame): Filtered raw SSL data used in the fit.
            - ssl_data_untilted (pandas.DataFrame): Tilt-corrected elevation data (via `ssl_results_elev_untilted`).

    Notes:
        - Tilt correction removes the effect of pitch, roll, and elevation offset from the measured elevation.
        - If the filtered dataset is empty, the function returns NaNs for all fit parameters.

    """

    # Get timestamps
    times = pd.to_datetime(ssl_data['Timestamp'])
    first_timestamp = times.iloc[0]
    last_timestamp = times.iloc[-1]

    # Drop NaN values and ensure distances are positive
    ssl_data.dropna(axis=0, inplace=True)
    if (ssl_data['Distance'] < 0).all():
        ssl_data['Distance'] = ssl_data['Distance'] * -1

    # Filter for cnr_max values
    if cnr_max:
        ssl_data = ssl_data.loc[ssl_data['CNR_max'] > cnr_max[0]]
        ssl_data = ssl_data.loc[ssl_data['CNR_max'] < cnr_max[1]]

    # Filter for growth filter
    if growth:
        ssl_data = ssl_data.loc[ssl_data['Growth'] > growth[0]]
        ssl_data = ssl_data.loc[ssl_data['Growth'] < growth[1]]

    # Exclude specified azimuth angles
    if azimuth_exclude is not None:
        if not isinstance(azimuth_exclude, (list, tuple, np.ndarray)):  # Falls keine Liste, Tuple oder np.array
            azimuth_exclude = [azimuth_exclude]  # In eine Liste umwandeln

        ssl_data = ssl_data.loc[~ssl_data['Azimuth'].isin(azimuth_exclude)]

    # Filter SSL data based on distance, elevation, and azimuth ranges
    if distance:
        ssl_data = ssl_data.loc[ssl_data['Distance'] > distance[0]]
        ssl_data = ssl_data.loc[ssl_data['Distance'] < distance[1]]
    if elevation:
        ssl_data = ssl_data.loc[ssl_data['Elevation'] > elevation[0]]
        ssl_data = ssl_data.loc[ssl_data['Elevation'] < elevation[1]]
    if azimuth:
        if azimuth[0] < azimuth[1]:
            ssl_data = ssl_data.loc[
                (ssl_data['Azimuth'] > azimuth[0]) & (ssl_data['Azimuth'] < azimuth[1])
                ]
        else:
            ssl_data = ssl_data.loc[
                (ssl_data['Azimuth'] > azimuth[0]) | (ssl_data['Azimuth'] < azimuth[1])
                ]

    # Apply optional distance correction
    correction = 0
    if distance_correct:
        correction = distance_correct
        ssl_data['Distance'] = ssl_data['Distance'] + distance_correct

    # Check whether data set is empty
    if ssl_data is None or len(ssl_data) == 0:
        print('------> Results for '+str(first_timestamp)+' are nan, because dataset is empty!')

        # empty data set
        ssl_data_untilted = None
        result_data = {
            'Timestamp': first_timestamp,
            'Last Timestamp': last_timestamp,
            'Distance Correction': correction,
            'Pitch': np.nan,
            'Roll': np.nan,
            'Lidar Height': np.nan,
            'Elevation Offset': np.nan,
            'Number of Values': np.nan,
            'Uncertainty Pitch': np.nan,
            'Uncertainty Roll': np.nan,
            'Uncertainty Lidar Height': np.nan,
            'Uncertainty Elevation Offset': np.nan,
            'RMSE': np.nan
        }

    else:
        # Determine the number of values
        count_data_length = len(ssl_data)

        # Calculate pitch, roll and elevation offset using the sin_cos_offset_fun function
        ssl_results, ssl_cov_matrix = curve_fit_offset(ssl_scan_results=ssl_data,
                                                       pitch_roll_offset_fun=extended_ssl_fun)

        # Calculation of uncertainty of parameters from the covariance matrix
        uncertainty = np.sqrt(np.diag(ssl_cov_matrix))

        # Combine timestamp and results in pandas DataFrame
        result_data = {
            'Timestamp': first_timestamp,
            'Last Timestamp': last_timestamp,
            'Distance Correction': correction,
            'Pitch': ssl_results[0],
            'Roll': ssl_results[1],
            'Lidar Height': ssl_results[2],
            'Elevation Offset': ssl_results[3],
            'Number of Values': count_data_length,
            'Uncertainty Pitch': uncertainty[0],
            'Uncertainty Roll': uncertainty[1],
            'Uncertainty Lidar Height': uncertainty[2],
            'Uncertainty Elevation Offset': uncertainty[3]
                }

        # Calculate tilt-corrected elevation angles and compute RMSE against theoretical model
        ssl_data_untilted = ssl_results_elev_untilted(ssl_results=ssl_data.copy(),
                                                      pitch=result_data['Pitch'],
                                                      roll=result_data['Roll'],
                                                      elevation_offset=result_data['Elevation Offset'])
        theo_elev_rmse = elev_depending_dist(dist_var=ssl_data_untilted['Distance'].values,
                                            h_lidar=result_data['Lidar Height'])
        rmse_elev = np.sqrt(np.mean((ssl_data_untilted['Elevation Untilted'].values - theo_elev_rmse) ** 2))

        # Update dictionary with RMSE
        result_data.update({"RMSE": rmse_elev})

    return result_data, ssl_data, ssl_data_untilted


# Function for plotting SSL results
def plot_ssl_results(ssl_results: dict, ssl_data: pd.DataFrame, ssl_data_untilted: pd.DataFrame, output_path: str):
    """
    Visualises Sea Surface Lidar (SSL) elevation results in the form of a correlation between
    elevation angle and distance to the sea surface, depending on the azimuth angle (colour axis).

    Args:
    ssl_results (dict):
        Dictionary containing calibration and correction parameters with
        the following required keys:

        - 'Pitch' (float): Pitch angle of the sensor in degrees.
        - 'Roll' (float): Roll angle of the sensor in degrees.
        - 'Elevation Offset' (float): Elevation offset angle in degrees.
        - 'Lidar Height' (float): Height of the lidar above sea level in meters.
        - 'RMSE' (float): Root Mean Square Error of the elevation correction
          in degrees.
        - 'Distance Correction' (bool or str): Indicator of whether a distance
          correction was applied.

    ssl_data (pandas.DataFrame):
        DataFrame containing distance-to-sea-surface data obtained from the
        CNR over-range analysis.

        Required columns:
        - 'Timestamp' (str): Measurement timestamp (YYYY-MM-DD HH:MM:SS).
        - 'Distance' (float): Distance to the sea surface in meters.
        - 'Elevation' (float): Measured elevation angle in degrees.
        - 'Azimuth' (float): Azimuth angle in degrees.

    ssl_data_untilted (pandas.DataFrame):
            DataFrame containing tilt-corrected ("untilted") elevation angles.
            The elevation angles have been corrected by removing the effect of the tilt
            induced by pitch, roll, and a fixed elevation offset, using the `ssl_results_elev_untilted` function.

        Required columns:
        - 'Distance' (float): Distance to the sea surface in meters.
        - 'Elevation Untilted' (float): Corrected elevation angle in degrees.
        - 'Azimuth' (float): Azimuth angle in degrees.

    output_path (str):
        Directory path where the generated plots will be saved.

    Returns:
        None: The function saves the generated plots to the specified output
        directory and displays them, but does not return any objects.

    """

    if ssl_data is None:
        print(f"---> Data ist None, no plots are created.")
    else:
        distance_correct = ssl_results['Distance Correction']

        # Calculate untilted elevation values depending on elevatio
        dist_start = ssl_data['Distance'].min() - 500
        dist_end = ssl_data['Distance'].max() + 500
        dist_array = np.array(np.arange(dist_start, dist_end, 1))
        plot_theo_elev = elev_depending_dist(dist_var=dist_array, h_lidar=ssl_results['Lidar Height'])

        # Extract and format timestamp information
        time = datetime.strptime(ssl_data['Timestamp'].iloc[0], '%Y-%m-%d %H:%M:%S')
        time_formatted = time.strftime('%Y_%m_%d_%H_%M')
        time_plot = time.strftime('%d.%m.%Y %H:%M')

        # Retrieve pitch, roll, elevation offset, lidar height, and RMSE values
        pitch = ssl_results['Pitch']
        roll = ssl_results['Roll']
        elevation_offset = ssl_results['Elevation Offset']
        lidar_height = ssl_results['Lidar Height']
        rmse_elev = ssl_results['RMSE']

        # Preparing the plot settings
        sns.set_theme(style="whitegrid")

        # Increase the number of bins and create a finer color palette
        num_bins = 36
        cmap = plt.cm.get_cmap('viridis', num_bins)

        # Normalize azimuth values for color mapping
        norm = plt.Normalize(ssl_data['Azimuth'].min(), ssl_data['Azimuth'].max())

        ################################################################################################################
        # PLOT 1: Uncorrected SSL data (Distance vs. Elevation, color-coded by Azimuth)
        ################################################################################################################
        fig, ax = plt.subplots(figsize=(10, 8))

        # Generate the scatter plot
        scatter = ax.scatter(ssl_data['Distance'],
                             ssl_data['Elevation'],
                             c=ssl_data['Azimuth'],
                             cmap=cmap, norm=norm)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Azimuth [°]', fontsize=20)

        # Label axes
        ax.set_ylabel('Elevation [°]', fontsize=20)
        ax.set_xlabel('Distance to Sea Surface [m]', fontsize=20)
        ax.tick_params(labelsize=20)

        # Set axis limits
        ax.set_xlim(dist_start, dist_end)
        ax.set_ylim(ssl_data['Elevation'].min()-0.1, ssl_data['Elevation'].max()+0.1)

        # Adjust tick label size for the colorbar
        cbar.ax.tick_params(labelsize=20)
        cbar.set_ticks(np.arange(0, 360, 45))

        # Place timestamp and distance correction information above the plot
        ax.text(0.95, 1.01, time_plot, fontsize=20, ha='right', va='bottom',
                color='dimgray', alpha=0.8, transform=ax.transAxes)
        ax.text(0.05, 1.01, f'distance correction {distance_correct}', fontsize=20, ha='left', va='bottom',
                color='dimgray', alpha=0.8, transform=ax.transAxes)

        # Adjust layout
        fig.tight_layout()

        # Save figure
        plotname = f'/SSL_results_{time_formatted}_correction_dist_{distance_correct}.png'
        output_filename = output_path + plotname
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')

        # Display plot
        plt.show()

        ################################################################################################################
        # PLOT 2: Corrected elevation angle data with the theoretical elevation curve
        ################################################################################################################
        fig, ax = plt.subplots(figsize=(10, 8))

        # Generate the scatter plot
        scatter = ax.scatter(ssl_data_untilted['Distance'], ssl_data_untilted['Elevation Untilted'],
                             c=ssl_data_untilted['Azimuth'],
                             cmap=cmap, norm=norm)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Azimuth [°]', fontsize=20)

        # Add theoretical abhängigkeit vom elevationswert zur Distance to sea surface
        ax.plot(dist_array, plot_theo_elev, color='red', linewidth=4, label=r'$\varphi_{Theo}$')

        # Label axes
        ax.set_ylabel('Elevation [°]', fontsize=20)
        ax.set_xlabel('Distance to Sea Surface [m]', fontsize=20)

        # Set axis limits
        ax.set_xlim(ssl_data_untilted['Distance'].min()-500, ssl_data_untilted['Distance'].max()+500)
        ax.set_ylim(ssl_data_untilted['Elevation Untilted'].min()-0.1, ssl_data_untilted['Elevation Untilted'].max()+0.1)
        ax.tick_params(labelsize=20)

        # Adjust tick label size for the colorbar
        cbar.ax.tick_params(labelsize=20)
        cbar.set_ticks(np.arange(0, 360, 45))

        # Add pitch, roll, elevation offset, lidar height, and RMSE to the legend
        ax.plot([], [], ' ', label=r'$\alpha$ = ' + f'{pitch:.2f}°')
        ax.plot([], [], ' ', label=r'$\beta$ = ' + f'{roll:.2f}°')
        ax.plot([], [], ' ', label=r'$\Phi$ = ' + f'{elevation_offset:.2f}°')
        ax.plot([], [], ' ', label=r'$\mathrm{h_{Lidar}}$ = ' + f'{lidar_height:.2f} m')
        ax.plot([], [], ' ', label=rf'{"RMSE = "} {rmse_elev:.2f}°')

        # Add legend
        ax.legend(fontsize=20, loc='lower right', frameon=True, fancybox=True, edgecolor='black', labelspacing=0.6,
                  handlelength=1.1)

        # Place timestamp and distance correction information above the plot
        ax.text(0.95, 1.01, time_plot, fontsize=20, ha='right', va='bottom',
                color='dimgray', alpha=0.8, transform=ax.transAxes)
        ax.text(0.05, 1.01, f'distance correction {distance_correct}', fontsize=20, ha='left', va='bottom',
                color='dimgray', alpha=0.8, transform=ax.transAxes)

        # Adjust layout
        fig.tight_layout()

        # Save figure
        plotname2 = f'/SSL_results_modelled_parameters_{time_formatted}_dist_correct_{distance_correct}.png'
        output_filename = output_path + plotname2
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')

        # Display plot
        plt.show()


