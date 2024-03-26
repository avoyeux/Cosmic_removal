import os
import pandas as pd
import numpy as np


main_path = '/home/avoyeux/Cosmic_removal/Cosmic_removal_errors'
mainfile_path = os.path.join(main_path, 'Alldata.csv')
pandas_alldata = pd.read_csv(mainfile_path)

avgs_used_images = []
time_intervals = []
tot_detections = []
tot_errors = []
tot_ratios = []
tot_detections_weighted = []
tot_errors_weighted = []
tot_ratios_weighted = []
pandas_intervals = pandas_alldata.groupby('Time interval')
for key, dataframe_time in pandas_intervals:
    dataframe_time = dataframe_time.reset_index(drop=True)

    exp_avgs_images = []
    exps = []
    exp_detections = []
    exp_errors = []
    exp_ratios = []
    exp_detections_weighted = []
    exp_errors_weighted = []
    exp_ratios_weighted = []
    pandas_exposures = dataframe_time.groupby('Exposure time')
    for key1, dataframe_exp in pandas_exposures:
        dataframe_exp = dataframe_exp.reset_index(drop=True)

        exp_avg_images = dataframe_exp['Nb of used images'].mean()
        exp_detection = dataframe_exp['Nb of detections'].sum()
        exp_error = dataframe_exp['Nb of errors'].sum()
        exp_detection_weighted = dataframe_exp['Weighted detections'].sum()
        exp_error_weighted = dataframe_exp['Weighted errors'].sum()
        if exp_detection != 0:
            exp_ratio = exp_error / exp_detection
            exp_ratio_weighted = exp_error_weighted / exp_detection_weighted
        else:
            exp_ratio = np.nan
            exp_ratio_weighted = np.nan

        exp_avgs_images.append(exp_avg_images)
        exps.append(key1)
        exp_detections.append(exp_detection)
        exp_errors.append(exp_error)
        exp_ratios.append(exp_ratio)
        exp_detections_weighted.append(exp_detection_weighted)
        exp_errors_weighted.append(exp_error_weighted)
        exp_ratios_weighted.append(exp_ratio_weighted)

    exp_name = f'Alldata_summary_inter{key}.csv'
    exp_path = os.path.join(main_path, f'Date_interval{key}')
    exp_dict = {'Time interval': np.full(len(exps), key), 'Exposure time': exps,
                'Avg nb of used images': exp_avgs_images, 'Nb detections': exp_detections, 'Nb errors': exp_errors,
                'Ratio': exp_ratios, 'Weighted detections': exp_detections_weighted,
                'Weighted errors': exp_errors_weighted, 'Weighted ratio': exp_ratios_weighted}
    nw_pandas_exp = pd.DataFrame(exp_dict)
    nw_pandas_exp.to_csv(os.path.join(exp_path, exp_name), index=False)

    # Global max simplified stats
    avg_used_images = dataframe_time['Nb of used images'].mean()
    tot_detection = dataframe_time['Nb of detections'].sum()
    tot_error = dataframe_time['Nb of errors'].sum()
    tot_detection_weighted = dataframe_time['Weighted detections'].sum()
    tot_error_weighted = dataframe_time['Weighted errors'].sum()
    if tot_detection != 0:
        tot_ratio = tot_error / tot_detection
        tot_ratio_weighted = tot_error_weighted / tot_detection_weighted
    else:
        tot_ratio = np.nan
        tot_ratio_weighted = np.nan

    avgs_used_images.append(avg_used_images)
    time_intervals.append(key)
    tot_detections.append(tot_detection)
    tot_errors.append(tot_error)
    tot_ratios.append(tot_ratio)
    tot_detections_weighted.append(tot_detection_weighted)
    tot_errors_weighted.append(tot_error_weighted)
    tot_ratios_weighted.append(tot_ratio_weighted)

inter_dict = {'Time interval': time_intervals, 'Avg used images': avgs_used_images, 'Nb detections': tot_detections,
              'Nb errors': tot_errors, 'Ratio': tot_ratios, 'Weighted detections': tot_detections_weighted,
              'Weighted errors': tot_errors_weighted, 'Weighted ratio': tot_ratios_weighted}
nw_pandas_inter = pd.DataFrame(inter_dict)
inter_name = f'Alldata_main_summary.csv'
nw_pandas_inter.to_csv(os.path.join(main_path, inter_name), index=False)
