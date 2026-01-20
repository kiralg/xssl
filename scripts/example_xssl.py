# Example for xssl

import xssl
import pandas as pd
from pathlib import Path


# Determine the project root directory (starting from scripts/example.py)
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Directories
folder_data = PROJECT_DIR / "data"
folder_out = PROJECT_DIR / "output"
folder_out.mkdir(parents=True, exist_ok=True)

# Files
file_bound = folder_data / "bounds_cnr_over_range.csv"
file_cnr = folder_data / "data_cnr_over_range_rhi1.csv"

# Filters for SSL method
dist_start = 500
dist_end = 4000
cnr_max_value0 = -30
cnr_max_value1 = 0
elev_start = -1.5
elev_end = -0.3
growth_start = 0.007
growth_end = 0.075
azimuth_exclude=[10, 15, 25, 30, 280, 290, 295, 300]
dist_correct = -37.5 #half the probe length of 75 m

if __name__ =='__main__':

    # Read data bounds
    data_bound = pd.read_csv(file_bound, delimiter=';')

    # Read data cnr
    data_cnr = xssl.read_data(file_cnr)

    # Determination of the distance to the water surface for each unique time step in the data
    ssl_data = xssl.wrapper_parallel_distance_analysis(data_cnr=data_cnr,
                                                       data_bound=data_bound,
                                                       file_out=folder_out,
                                                       cnr_threshold=-30,
                                                       show_plot=True,
                                                       num_cpu=6)

    # Use Extended SSL method to get pitch, roll and elevation offset
    ssl_result, ssl_data_filtered, ssl_data_untilted = xssl.ssl_wrapper(ssl_data=ssl_data,
                                                        distance=[dist_start, dist_end],
                                                        elevation=[elev_start, elev_end],
                                                        growth=[growth_start, growth_end],
                                                        cnr_max=[cnr_max_value0, cnr_max_value1],
                                                        distance_correct=dist_correct,
                                                        azimuth_exclude=azimuth_exclude)
    # Plot results
    xssl.plot_ssl_results(ssl_results=ssl_result, ssl_data=ssl_data_filtered, ssl_data_untilted=ssl_data_untilted,
                          output_path=folder_out)

    # Save results to a csv file
    df = pd.DataFrame([ssl_result])
    filename = f"SSL_results_" + df.loc[0, 'Timestamp'].strftime('%Y_%m_%d_%H_%M_%S') + f"_cor{dist_correct}.csv"
    output_filename = folder_out / filename
    df.to_csv(output_filename,
              sep=";",
              index=False)
