# IMPORTS
import os
import sys
import common
from multiprocessing import Process
import warnings
import numpy as np
import pandas as pd
from simple_decorators import decorators
import matplotlib as mpl
from astropy.io import fits
import multiprocessing as mp
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from dateutil.parser import parse as parse_date


# Creating a class for the cosmicremoval
class Cosmicremoval_class:
    # Finding the L1 darks
    cat = common.SpiceUtils.read_spice_uio_catalog()
    filters = cat.STUDYDES.str.contains('dark') & (cat['LEVEL'] == 'L1')
    res = cat[filters]

    def __init__(self, processes=7, chunk_nb=4, coefficient=6, min_filenb=30, months_interval=200, min_files=12, bins=2000):
        # Inputs
        self.processes = processes
        self.chunk_nb = chunk_nb
        self.coef = coefficient
        self.min_filenb = min_filenb
        self.months_interval = months_interval  # time interval to consider in months
        self.min_files = min_files
        self.bins = bins

        # Miscellaneous
        self.ref_pixels = list(map(int, np.linspace(0, 1023, 2)))  # reference pixels for the init madmode calculations

        # Code functions
        self.exposures = self.Exposure()

    ################################################ INITIAL functions #################################################
    def Paths(self, exposure='none', detector='none'):
        """Function to create all the different paths. Lots of if statements to be able to add files where ever I want
        """
        main_path = os.path.join(os.getcwd(), f'Temporal_coef{self.coef}_{self.months_interval}months_bins{self.bins}_'
                                              f'newtry')

        if exposure != 'none':
            exposure_path = os.path.join(main_path, f'Exposure{exposure}')

            if detector != 'none':
                detector_path = os.path.join(exposure_path, f'Detector{detector}')
                # Main paths
                initial_paths = {'Main': main_path, 'Exposure': exposure_path, 'Detector': detector_path}
                # Secondary paths
                directories = ['Darks', 'Masks', 'Histograms', 'Statistics', 'Temperatures', 'Dates', 'Medians',
                               'Special histograms', 'Results']
                paths = {}
                for directory in directories:
                    path = os.path.join(detector_path, directory)
                    paths[directory] = path
            else:
                initial_paths = {'Main': main_path, 'Exposure': exposure_path}
                paths = {}
        else:
            initial_paths = {'Main': main_path}
            paths = {}

        all_paths = {}
        for d in [initial_paths, paths]:
            for key, path in d.items():
                os.makedirs(path, exist_ok=True)
            all_paths.update(d)

        return all_paths

    def Exposure(self):
        """Function to find the different exposure times in the SPICE catalogue"""

        # Getting the exposure times and nb of occurrences
        exposure_counts = Counter(Cosmicremoval_class.res.XPOSURE)
        exposure_weighted = np.array(list(exposure_counts.items()))

        # Printing the values
        for loop in range(len(exposure_weighted)):
            print(f'For exposure time {exposure_weighted[loop, 0]}s there are {int(exposure_weighted[loop, 1])} darks.')

        # Keeping the exposure times with enough darks
        occurrences_filter = exposure_weighted[:, 1] > self.min_filenb
        exposure_used = exposure_weighted[occurrences_filter][:, 0]
        print(f'\033[93mExposure times with less than \033[1m{self.min_filenb}\033[0m\033[93m darks are not kept.'
              f'\033[0m')
        print(f'\033[33mExposure times kept are {exposure_used}\033[0m')

        # Saving exposure stats
        paths = self.Paths()
        csv_name = 'All_exposuretimes.csv'
        exp_dict = {'Exposure time (s)': exposure_weighted[:, 0], 'Number of darks': exposure_weighted[:, 1]}
        pandas_dict = pd.DataFrame(exp_dict)
        sorted_dict = pandas_dict.sort_values(by='Exposure time (s)')
        sorted_dict.to_csv(os.path.join(paths['Main'], csv_name), index=False)

        return exposure_used

    def Images_all(self, exposure):
        """Function to get, for a certain exposure time and detector nb, the corresponding images, distance to sun,
        temperature array and the corresponding filenames"""

        # Filtering the data by exposure time
        filter = (Cosmicremoval_class.res.XPOSURE == exposure)
        res = Cosmicremoval_class.res[filter]

        # All filenames
        filenames = np.array(list(res['FILENAME']))

        # Variable initialisation
        a = 0
        left_nb = 0
        weird_nb = 0
        all_images = []
        all_dsun = []
        all_dates = []
        all_temps = []
        all_files = []
        for loop, file in enumerate(filenames):
            # Opening the files
            hdul = fits.open(common.SpiceUtils.ias_fullpath(file))

            if hdul[0].header['BLACKLEV'] == 1:  # For on-board bias subtraction images
                left_nb += 1
                continue
            OBS_DESC = hdul[0].header['OBS_DESC'].lower()
            if 'glow' in OBS_DESC:  # Weird "glow" darks
                weird_nb += 1
                continue

            temp1 = hdul[0].header['T_SW']
            temp2 = hdul[0].header['T_LW']

            if temp1 > 0 and temp2 > 0:
                a += 1
                continue
            elif temp1 > 0:
                print(f"weird, filename{file}")
                continue
            elif temp2 > 0:
                print(f"weird, filename{file}")
                continue

            # Needed stats
            dist_SUN = hdul[0].header['DSUN_OBS']
            date = hdul[0].header['DATE-BEG']

            images = []
            dsun = []
            temps = []
            dates = []
            files = []
            for detector in range(2):
                if detector == 0:
                    temp = temp1
                else:
                    temp = temp2
                data = hdul[detector].data
                images.append(np.double(data[0, :, :, 0]))
                dsun.append(dist_SUN)
                temps.append(temp)
                dates.append(date)
                files.append(file)
            all_images.append(images)
            all_dsun.append(dsun)
            all_temps.append(temps)
            all_dates.append(dates)
            all_files.append(files)
        all_images = np.array(all_images)
        all_dsun = np.array(all_dsun)
        all_dates = np.array(all_dates)
        all_temps = np.array(all_temps)
        all_files = np.array(all_files)

        if a != 0:
            print(f'\033[31mExp{exposure} -- Tot nb files with high temp: {a}\033[0m')
        if left_nb != 0:
            print(f'\033[31mExp{exposure} -- Tot nb files with bias subtraction: {left_nb}\033[0m')
        if weird_nb != 0:
            print(f'\033[31mExp{exposure} -- TOT nb of "weird" files: {weird_nb}\033[0m')
        print(f'Exp{exposure} -- Nb of "usable" files: {len(all_files)}')
        if len(all_files) == 0:
            print(f'\033[91m ERROR: NO "USABLE" ACQUISITIONS - STOPPING THE RUN\033[0m')
            sys.exit()

        return all_images, all_dates, all_dsun, all_temps, all_files

    ############################################### CALCULATION functions ##############################################
    @decorators.running_time
    def Multiprocess(self):
        """Function for multiprocessing if self.processes > 1. No multiprocessing done otherwise."""

        # Choosing to multiprocess or not
        if self.processes > 1:
            print(f'self.processes is {self.processes}')
            args = [(exposure,) for exposure in self.exposures]
            pool = mp.Pool(processes=self.processes)
            pool.starmap(self.Main, args)
            pool.close()
            pool.join()
        else:
            for exposure in self.exposures:
                self.Main(exposure)

    @decorators.running_time
    def Main(self, exposure):
        print(f'The process id is {os.getpid()}')
        # MAIN LOOP
        # Initialisation of the stats for csv file saving
        data_pandas_exposure = pd.DataFrame()
        # data_pandas_std = pd.DataFrame()
        # data_pandas_mad = pd.DataFrame()
        all_images, all_date, all_dsun, all_temp, all_filenames = self.Images_all(exposure)

        if len(all_filenames) < self.min_filenb:
            print(f'\033[91m Exp{exposure} -- Less than {self.min_filenb} usable files. '
                  f'Changing exposure times.\033[0m')
            return

        for detector in range(2):
            paths = self.Paths(exposure=exposure, detector=detector)

            images, date, dsun = all_images[:, detector], all_date[:, detector], all_dsun[:, detector]
            temp, filenames = all_temp[:, detector], all_filenames[:, detector]  # TODO: no need for file name, dsun and dates

            # # Saving these stats in a csv file
            # new_dict = {'Nb of used files': np.full((2,), len(filenames))}
            # pandas_new = pd.DataFrame(new_dict)
            # csv_name = f'Info_nbfiles{exposure}.csv'
            # pandas_new.to_csv(os.path.join(paths['Main'], csv_name), index=False)

            # MULTIPLE DARKS analysis
            same_darks, positions = self.Samedarks(filenames)
            data_pandas_detector = pd.DataFrame()

            # # Saving the stats in csv files
            # pandas_std, pandas_mad = self.Percentages_stdnmad(exposure, detector, chunks, chunks_mad, chunks_mode)
            # std_name = f'Info_stdpercentages_exp{exposure}_det{detector}.csv'
            # mad_name = f'Info_madpercentages_exp{exposure}_det{detector}.csv'
            # pandas_std.to_csv(os.path.join(paths['Detector'], std_name), index=False)
            # pandas_mad.to_csv(os.path.join(paths['Detector'], mad_name), index=False)
            # print(f'Exp{exposure}_det{detector} -- Percentages saved to csv file.')

            print(f'Exp{exposure}_det{detector} -- Starting chunks.')
            for SPIOBSID, files in same_darks.items():
                if len(files) < 3:
                    continue

                data, mads, modes, masks = self.time_interval(exposure, detector, filenames, files, images, positions,
                                                            SPIOBSID)  # TODO: here
                if len(data) == 0:
                    continue

                # Error calculations
                nw_masks, detections, errors, ratio, weights_tot, weights_error, weights_ratio = self.Stats(data, masks,
                                                                                                            modes)

                # # Saving the stats in a csv file
                data_pandas = self.Unique_datadict(exposure, detector, files, mads, modes, detections, errors, ratio,
                                                   weights_tot, weights_error, weights_ratio)
                csv_name = f'Info_for_ID{SPIOBSID}.csv'
                data_pandas.to_csv(os.path.join(paths['Statistics'], csv_name), index=False)
                data_pandas_detector = pd.concat([data_pandas_detector, data_pandas])
            #print(f'Exp{exposure}_det{detector} -- Chunks finished and Median plotting done.')

            # Combining the dictionaries
            # data_pandas_std = pd.concat([data_pandas_std, pandas_std])
            # data_pandas_mad = pd.concat([data_pandas_mad, pandas_mad])
            data_pandas_exposure = pd.concat([data_pandas_exposure, data_pandas_detector])

        # Saving a csv file for each exposure time
        # std_name = f'Info_stdpercentages_exp{exposure}.csv'
        # mad_name = f'Info_madpercentages_exp{exposure}.csv'
        csv_name = f'Info_for_exp{exposure}.csv'
        # data_pandas_std.to_csv(os.path.join(paths['Exposure'], std_name), index=False)
        # data_pandas_mad.to_csv(os.path.join(paths['Exposure'], mad_name), index=False)
        data_pandas_exposure.to_csv(os.path.join(paths['Exposure'], csv_name), index=False)
        #print(f'Exp{exposure} -- CSV files created')

    def time_interval(self, exposure, detector, filenames, files, images, positions, SPIOBSID):
        first_filename = files[0]
        name_dict = common.SpiceUtils.parse_filename(first_filename)
        date = parse_date(name_dict['time'])

        date_interval = int(self.months_interval / 2)
        year_max = date.year  # TODO: need to add the value to the self.  (????)
        year_min = date.year
        month_max = date.month + date_interval
        month_min = date.month - date_interval

        if month_max > 12:
            year_max += (month_max - 1) // 12
            month_max = month_max % 12
        if month_min < 1:
            year_min -= (abs(month_min) // 12) + 1
            month_min = 12 - (abs(month_min) % 12)

        date_max = f'{year_max:04d}{month_max:02d}{date.day:02d}T{date.hour:02d}{date.minute:02d}{date.second:02d}'
        date_min = f'{year_min:04d}{month_min:02d}{date.day:02d}T{date.hour:02d}{date.minute:02d}{date.second:02d}'

        mads = []
        modes = []
        masks = []
        data = []
        for file in files:
            used_images = []
            rank = -1
            for loop2, file2 in enumerate(filenames):
                name_dict = common.parse_date(file2)
                date = name_dict['time']
                if date < date_min or date > date_max:
                    continue
                if file2 in files and file2 != file:
                    continue
                rank += 1
                if file2 == file:
                    position = rank
                used_images.append(images[loop2])
            used_images = np.array(used_images)
            mad, mode, mask = self.Chunks_func(used_images)

            mads.append(mad)
            modes.append(mode)
            masks.append(mask[position])
            data.append(used_images[position])
        mads = np.array(mads)
        modes = np.array(modes)
        masks = np.array(masks)
        data = np.array(data)
        return data, mads, modes, masks

    def Percentages_stdnmad(self, exposure, detector, data, mad, mode):
        std_kept = np.zeros_like(data)
        mad_kept = np.zeros_like(data)
        std_filter = (data > mode - 1 * np.std(data, axis=0)) & (data < mode + np.std(data, axis=0))
        mad_filter = (data > mode - 1 * mad) & (data < mode + mad)
        std_kept[std_filter] = 1
        mad_kept[mad_filter] = 1

        std_percentage = np.sum(std_kept, axis=0) / data.shape[0]
        mad_percentage = np.sum(mad_kept, axis=0) / data.shape[0]

        rows = np.arange(0, std_percentage.shape[0])
        init_dict = {'Exposure': exposure, 'Detector': detector, 'Rows': rows}
        pandas_init = pd.DataFrame(init_dict)

        std_dict = {f'Column {i}': std_percentage[:, i] for i in range(std_percentage.shape[1])}
        mad_dict = {f'Column {i}': mad_percentage[:, i] for i in range(mad_percentage.shape[1])}

        panda_std, panda_mad = pd.DataFrame(std_dict), pd.DataFrame(mad_dict)
        pandas_std = pd.concat([pandas_init, panda_std], axis=1)
        pandas_mad = pd.concat([pandas_init, panda_mad], axis=1)
        return pandas_std, pandas_mad

    def Unique_datadict(self, exposure, detector, files, mads, modes, detections, errors, ratio, weights_tot,
                        weights_error, weights_ratio):
        """Function to create a dictionary containing some useful information on each exposure times. This is
         done to save the stats in a csv file when the code finishes running."""

        # Initialisation
        name_dict = common.SpiceUtils.parse_filename(files[0])
        date = parse_date(name_dict['time'])

        # Creation of the stats
        group_nb = len(files)
        group_date = f'{date.year:04d}{date.month:02d}{date.day:02d}'
        SPIOBSID = name_dict['SPIOBSID']
        tot_detection = np.sum(detections)
        tot_error = np.sum(errors)
        if tot_detection != 0:
            tot_ratio = tot_error / tot_detection
        else:
            tot_ratio = np.nan
        # Corresponding lists or arrays
        a, b, c, d, e, f, g, h = np.full((group_nb, 8), [exposure, detector, group_date, group_nb, tot_detection,
                                                         tot_error, tot_ratio, SPIOBSID]).T

        data_dict = {'Exposure time': a, 'Detector': b, 'Group date': c, ' Nb of files in group': d,
                     'Tot nb of detections': e, 'Tot nb of errors': f, 'Ratio errors/detections': g,
                     'Filename': files, 'SPIOBSID': h, 'Average Mode': np.mean(modes),
                     'Average mode absolute deviation': np.mean(mads), 'Nb of detections': detections,
                     'Nb of errors': errors, 'Ratio': ratio, 'Weighted detections': weights_tot,
                     'Weighted errors': weights_error, 'Weighted ratio': weights_ratio}

        Pandasdata = pd.DataFrame(data_dict)
        return Pandasdata

    def Madmode_list_ref(self, images):
        """Function to get the mad and mode values for the reference pixels and the main and left lists"""

        # Variable initialisation
        ref_madarray = np.zeros_like(images[0, :, :])
        ref_modearray = np.zeros_like(ref_madarray)
        for r in self.ref_pixels:
            for c in self.ref_pixels:
                # Variable initialisation
                data = images[:, r, c]
                bins = self.Bins(data)

                # Creating a histogram
                hist, bin_edges = np.histogram(data, bins=bins)
                max_bin_index = np.argmax(hist)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                mode = bin_centers[max_bin_index]

                # Determination of the mode absolute deviation
                mad = np.mean(np.abs(data - mode))
                ref_madarray[r, c] = mad
                ref_modearray[r, c] = mode
        return ref_madarray, ref_modearray

    def Chunk_madmodemask(self, chunk):
        """Function to calculate the mad, mode and mask for a given chunk
        (i.e. spatial chunk with all the temporal values)"""

        # Variable initialisation
        mad_array = np.zeros_like(chunk[0, :, :])
        mode_array = np.zeros_like(mad_array)
        # masks = np.zeros_like(chunk, dtype='bool')
        masks = np.zeros_like(chunk, dtype='bool')
        for r in range(chunk.shape[1]):
            for c in range(chunk.shape[2]):
                # Variable initialisation
                data = chunk[:, r, c]
                bins = self.Bins(data)

                # Creating a histogram
                hist, bin_edges = np.histogram(data, bins=bins)
                max_bin_index = np.argmax(hist)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                mode = bin_centers[max_bin_index]

                # Determination of the mode absolute deviation
                mad = np.mean(np.abs(data - mode))
                mad_array[r, c] = mad
                mode_array[r, c] = mode
        # Mad clipping to get the chunk specific mask
        masks_filter = chunk > self.coef * mad_array + mode_array
        masks[masks_filter] = True
        return mad_array, mode_array, masks  # these are all the values for each chunk

    def Chunks_func(self, images):
        """Function to fusion all the mode, mad and masks values from all the chunks"""

        # Variable initialisation
        chunks = np.split(images, self.chunk_nb, axis=1)

        # Creating the chunk specific mad, mode and mask arrays
        chunks_mad, chunks_mode, chunks_mask = self.Chunk_madmodemask(chunks[0])
        for loop in range(1, len(chunks)):
            chunk = chunks[loop]
            chunk_mad, chunk_mode, chunk_mask = self.Chunk_madmodemask(chunk)

            # Saving the data
            chunks_mad = np.concatenate((chunks_mad, chunk_mad), axis=0)  # (1024*1024 array)
            chunks_mode = np.concatenate((chunks_mode, chunk_mode), axis=0)
            chunks_mask = np.concatenate((chunks_mask, chunk_mask), axis=1)  # TODO: need to check if it axis 0 or 1
        return chunks_mad, chunks_mode, chunks_mask

    def Samedarks(self, filenames):
        # Dictionaries initialisation
        same_darks = {}
        positions = {}
        for loop, file in enumerate(filenames):
            d = common.SpiceUtils.parse_filename(file)
            SPIOBSID = d['SPIOBSID']
            if SPIOBSID not in same_darks:
                same_darks[SPIOBSID] = []
                positions[SPIOBSID] = []
            same_darks[SPIOBSID].append(file)
            positions[SPIOBSID].append(loop)
        return same_darks, positions

    def Stats(self, data, masks, modes):
        """Function to calculate some stats to have an idea of the efficacy of the method. The output is a set of masks
        giving the positions where the method outputted a worst result than the initial image"""

        # Initialisation
        nw_data = np.copy(data)
        data_med = np.median(data, axis=0)
        meds_dif = data - data_med

        # Difference between the end result and the initial one
        nw_data[masks] = modes[masks]
        nw_meds_dif = nw_data - data_med

        # Creating a new set of masks that shows where the method made an error
        nw_masks = np.zeros_like(masks, dtype='bool')
        filters = np.abs(nw_meds_dif) > np.abs(meds_dif)
        nw_masks[filters] = True

        ### MAIN STATS
        # Initialisation of the corresponding matrices
        weights_errors = np.zeros_like(data)
        weights_errors[nw_masks] = np.abs(np.abs(meds_dif[nw_masks]) - np.abs(nw_meds_dif[nw_masks]))
        weights_tots = np.abs(np.abs(meds_dif) - np.abs(nw_meds_dif))
        # Calculating the number of detections and errors per dark
        detections = np.sum(masks, axis=(1, 2))
        errors = np.sum(nw_masks, axis=(1, 2))
        # Calculating the "weighted error"
        weights_error = np.sum(weights_errors, axis=(1, 2))
        weights_tot = np.sum(weights_tots, axis=(1, 2))
        # Calculating the ratios
        if 0 not in detections:  # If statements seperates two cases (with or without detections)
            ratio = errors / detections
            weights_ratio = weights_error / weights_tot
        else:
            ratio = []
            weights_ratio = []
            for loop, detval in enumerate(detections):
                if detval != 0:
                    ratio1 = errors[loop] / detval
                    weights_ratio1 = weights_error[loop] / weights_tot[loop]
                else:
                    ratio1 = np.nan
                    weights_ratio1 = np.nan
                ratio.append(ratio1)
                weights_ratio.append(weights_ratio1)
            ratio = np.array(ratio)
            weights_ratio = np.array(weights_ratio)
        return nw_masks, detections, errors, ratio, weights_tot, weights_error, weights_ratio

    ################################################ PLOTTING functions ################################################
    def Bins(self, data):
        """Small function to calculate the appropriate bin count"""
        val_range = np.max(data) - np.min(data)
        bins = int(len(data) * val_range / self.bins)  #was 500 before
        # bins = np.array(range(int(np.min(data)), int(np.max(data)) + 2, self.bins))
        if isinstance(bins, int):
            if bins < 10:
                bins = 10

        elif isinstance(bins, np.ndarray):
            if len(bins) < 10:
                bins = 10
        return bins

if __name__ == '__main__':
    mpl.rcParams['figure.figsize'] = (8, 8)

    warnings.filterwarnings('ignore', category=mpl.MatplotlibDeprecationWarning)
    test = Cosmicremoval_class(min_filenb=30)
    test.Multiprocess()
    warnings.filterwarnings("default", category=mpl.MatplotlibDeprecationWarning)

