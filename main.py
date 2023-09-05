import os
import pandas as pd
import numpy as np

def func(exposures):
    init_path = os.path.join(os.getcwd(), 'Temporal_coef6_8months_bins1000_nw2')
    detections = []
    errors = []
    weighted_errors = []
    weighted_detections = []

    for exp in exposures:
        csv_path = os.path.join(init_path, f'Exposure{exp}', f'Info_for_exp{exp}.csv')

        panda_data = pd.read_csv(csv_path)

        nb_detections = panda_data['Nb of detections']
        nb_errors = panda_data['Nb of errors']
        nb_weightederrors = panda_data['Weighted errors']
        nb_weighteddetections = panda_data['Weighted detections']

        tot_detections = np.sum(nb_detections)
        tot_errors = np.sum(nb_errors)
        tot_weighteddet = np.sum(nb_weighteddetections)
        tot_weightederr = np.sum(nb_weightederrors)

        detections.append(tot_detections)
        errors.append(tot_errors)
        weighted_errors.append(tot_weightederr)
        weighted_detections.append(tot_weighteddet)

    lists = [detections, errors, weighted_detections, weighted_errors]
    detections, errors, weighted_detections, weighted_errors = [np.array(lst) for lst in lists]

    ratio = errors / detections
    weighted_ratio = weighted_errors / weighted_detections

    data_dict = {'Exposure times (s)': exposures, 'Detections': detections, 'Errors': errors, 'Ratio': ratio,
                 'Weighted detections': weighted_detections, 'Weighted errors': weighted_errors,
                 'Weighted ratio': weighted_ratio}
    panda_data = pd.DataFrame(data_dict)
    panda_data = panda_data.sort_values(by='Exposure times (s)')

    panda_filenb = pd.read_csv(os.path.join(init_path, 'All_exposuretimes.csv'))
    column_to_add_tuple = (panda_filenb['Exposure time (s)'], panda_filenb['Number of darks'])

    nb_darks = []
    for exp in panda_data['Exposure times (s)']:
        for loop in range(len(column_to_add_tuple[0])):
            if exp == column_to_add_tuple[0][loop]:
                nb_darks.append(column_to_add_tuple[1][loop])


    panda_data['Number of darks'] = nb_darks

    csv_name = f'Allexposures_ratiosummary.csv'
    panda_data.to_csv(os.path.join(init_path, csv_name), index=False)

exposures = [0.1, 4.6, 9.6, 19.6, 29.6, 59.6, 89.6]

func(exposures)
