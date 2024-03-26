# IMPORTS
import os
import sys
import common
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import multiprocessing as mp
import matplotlib.pyplot as plt

from astropy.io import fits
from collections import Counter
from typeguard import typechecked
from simple_decorators import decorators
from dateutil.parser import parse as parse_date


# Creating a class for the cosmicremoval
class Cosmicremoval_class:
    """
    To get the SPICE darks, seperate them depending on exposure time, do a temporal analysis of the detector counts for each 
    pixels, flag any values that are too far from the distribution as cosmic rays to then replace them by the mode of the 
    detector count for said pixel.
    """

    # Finding the L1 darks
    cat = common.SpiceUtils.read_spice_uio_catalog()
    filters = cat.STUDYDES.str.contains('dark') & (cat['LEVEL'] == 'L1')
    res = cat[filters]

    @typechecked
    def __init__(self, processes: int = 90, coefficient: int | float = 6, min_filenb: int = 20, set_min: int = 4,
                time_intervals: np.ndarray = np.array([4, 40]), bins: int = 5, stats: bool = True, plots: bool = True):
        
        # Arguments
        self.processes = processes
        self.coef = coefficient
        self.min_filenb = min_filenb
        self.set_min = set_min
        self.stats = stats  # Bool to know if stats will be saved
        self.plots = plots  # boolean to know if plots will be saved
        # self.time_intervals = time_intervals
        self.time_intervals = np.arange(2, 20, 4)
        self.bins = bins

        # Code functions
        self.exposures = self.Exposure()

    ################################################ INITIAL functions #################################################
    def Paths(self, time_interval=-1, exposure=-1, detector=-1):
        """
        Function to create all the different paths. Lots of if statements to be able to add files where ever I want
        """

        main_path = os.path.join(os.getcwd(), 'Cosmic_removal_errors2')

        if time_interval != -1:
            time_path = os.path.join(main_path, f'Date_interval{time_interval}')

            if exposure != -1:
                exposure_path = os.path.join(time_path, f'Exposure{exposure}')

                if detector != -1:
                    detector_path = os.path.join(exposure_path, f'Detector{detector}')
                    # Main paths
                    initial_paths = {'Main': main_path, 'Time interval': time_path, 'Exposure': exposure_path,
                                     'Detector': detector_path}
                    # Secondary paths
                    directories = ['Histograms', 'Statistics', 'Special histograms']
                    paths = {}
                    for directory in directories:
                        path = os.path.join(detector_path, directory)
                        paths[directory] = path
                else:
                    initial_paths = {'Main': main_path, 'Time interval': time_path, 'Exposure': exposure_path}
                    paths = {}
            else:
                initial_paths = {'Main': main_path, 'Time interval': time_path}
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
        """
        Function to find the different exposure times in the SPICE catalogue.
        """

        # Getting the exposure times and nb of occurrences
        exposure_counts = Counter(Cosmicremoval_class.res.XPOSURE)
        exposure_weighted = np.array(list(exposure_counts.items()))

        # Printing the values
        [print(f'For exposure time {exposure[0]}s there are {int(exposure[1])} darks.') for exposure in exposure_weighted]

        # Keeping the exposure times with enough darks
        occurrences_filter = exposure_weighted[:, 1] > self.min_filenb
        exposure_used = exposure_weighted[occurrences_filter][:, 0]
        print(f'\033[93mExposure times with less than \033[1m{self.min_filenb}\033[0m\033[93m darks are not kept.'
              f'\033[0m')
        print(f'\033[33mExposure times kept are {exposure_used}\033[0m')

        if self.stats:
            # Saving exposure stats
            paths = self.Paths()
            csv_name = 'Nb_of_darks.csv'
            exp_dict = {'Exposure time (s)': exposure_weighted[:, 0], 'Total number of darks': exposure_weighted[:, 1]}
            pandas_dict = pd.DataFrame(exp_dict)
            sorted_dict = pandas_dict.sort_values(by='Exposure time (s)')
            sorted_dict.to_csv(os.path.join(paths['Main'], csv_name), index=False)

        return exposure_used

    def Images_all(self, exposure):
        """
        Function to get, for a certain exposure time and detector nb, the corresponding images, distance to sun,
        temperature array and the corresponding filenames.
        """

        # Filtering the data by exposure time
        filters = (Cosmicremoval_class.res.XPOSURE == exposure)
        res = Cosmicremoval_class.res[filters]

        # All filenames
        filenames = np.array(list(res['FILENAME']))

        # Variable initialisation
        a = 0
        left_nb = 0
        weird_nb = 0
        all_images = []
        all_files = []
        for files in filenames:
            # Opening the files
            hdul = fits.open(common.SpiceUtils.ias_fullpath(files))

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

            images = []
            for detector in range(2):
                data = hdul[detector].data
                images.append(np.double(data[0, :, :, 0]).astype('int32'))

            all_images.append(images)
            all_files.append(files)
        all_images = np.array(all_images)
        all_files = np.array(all_files)

        if a != 0:
            print(f'\033[31mExp{exposure} -- Tot nb files with high temp: {a}\033[0m')
        if left_nb != 0:
            print(f'\033[31mExp{exposure} -- Tot nb files with bias subtraction: {left_nb}\033[0m')
        if weird_nb != 0:
            print(f'\033[31mExp{exposure} -- Tot nb of "weird" files: {weird_nb}\033[0m')
        print(f'Exp{exposure} -- Nb of "usable" files: {len(all_files)}')
        if len(all_files) == 0:
            print(f'\033[91m ERROR: NO "USABLE" ACQUISITIONS - STOPPING THE RUN\033[0m')
            sys.exit()
        return all_images, all_files

    ############################################### CALCULATION functions ##############################################
    @decorators.running_time
    def Multiprocess(self):
        """
        Function for multiprocessing if self.processes > 1. No multiprocessing done otherwise.
        """

        # Choosing to multiprocess or not
        if self.processes > 1:
            print(f'Number of used processes is {self.processes}')
            args = list(itertools.product(self.time_intervals, self.exposures))
            pool = mp.Pool(processes=self.processes)
            data_pandas_interval = pool.starmap(self.Main, args)
            pool.close()
            pool.join()

            if self.stats:
                # Saving the data in csv files
                self.Saving_csv(data_pandas_interval)

        else:
            data_pandas_all = pd.DataFrame()

            for time_inter in self.time_intervals:
                data_pandas_interval = pd.DataFrame()

                for exposure in self.exposures:
                    data_pandas_exposure = self.Main(time_inter, exposure)

                    if not self.stats:
                        continue
                    paths = self.Paths(time_interval=time_inter, exposure=exposure)
                    data_pandas_interval = pd.concat([data_pandas_interval, data_pandas_exposure], ignore_index=True)
                    pandas_name0 = f'Alldata_inter{time_inter}_exp{exposure}.csv'
                    data_pandas_exposure.to_csv(os.path.join(paths['Exposure'], pandas_name0), index=False)

                if not self.stats:
                    continue
                data_pandas_all = pd.concat([data_pandas_all, data_pandas_interval], ignore_index=True)
                pandas_name1 = f'Alldata_inter{time_inter}.csv'
                data_pandas_interval.to_csv(os.path.join(paths['Time interval'], pandas_name1), index=False)

            if self.stats:
                pandas_name = 'Alldata.csv'
                data_pandas_all.to_csv(os.path.join(paths['Main'], pandas_name), index=False)

    def Saving_csv(self, data_list):
        args = list(itertools.product(self.time_intervals, self.exposures))
        last_time = 0
        first_try = 0

        for loop, pandas_dict in enumerate(data_list):
            indexes = args[loop]
            paths = self.Paths(time_interval=indexes[0], exposure=indexes[1])

            # Saving a csv file for each exposure time
            csv_name = f'Alldata_inter{indexes[0]}_exp{indexes[1]}.csv'
            pandas_dict.to_csv(os.path.join(paths['Exposure'], csv_name), index=False)
            print(f'Inter{indexes[0]}_exp{indexes[1]} -- CSV files created', flush=True)

            if indexes[0] == last_time:
                pandas_inter = pd.concat([pandas_inter, pandas_dict], ignore_index=True)
                if indexes == args[-1]:
                    pandas_name0 = f'Alldata_inter{indexes[0]}.csv'
                    pandas_inter.to_csv(os.path.join(paths['Time interval'], pandas_name0), index=False)
                    print(f'Inter{indexes[0]} -- CSV files created', flush=True)
            else:
                if first_try != 0:
                    paths = self.Paths(time_interval=last_time)
                    pandas_name0 = f'Alldata_inter{last_time}.csv'
                    pandas_inter.to_csv(os.path.join(paths['Time interval'], pandas_name0), index=False)
                    print(f'Inter{indexes[0]} -- CSV files created', flush=True)
                first_try = 1
                last_time = indexes[0]
                pandas_inter = pandas_dict

        data_list = pd.concat(data_list, ignore_index=True)
        pandas_name = 'Alldata.csv'
        data_list.to_csv(os.path.join(paths['Main'], pandas_name), index=False)

    @decorators.running_time
    def Main(self, time_interval, exposure):
        print(f'The process id is {os.getpid()}')
        time_interval = int(time_interval)
        # MAIN LOOP
        # Initialisation of the stats for csv file saving
        data_pandas_exposure = pd.DataFrame()
        all_images, filenames = self.Images_all(exposure)

        if len(filenames) < self.min_filenb:  # TODO: maybe need to change this if I take the med or mean
            print(f'\033[91mInter{time_interval}_exp{exposure} -- Less than {self.min_filenb} usable files. '
                  f'Changing exposure times.\033[0m')
            return

        # MULTIPLE DARKS analysis
        same_darks, positions = self.Samedarks(filenames)
        for detector in range(2):
            paths = self.Paths(time_interval=time_interval, exposure=exposure, detector=detector)
            images = all_images[:, detector]
            data_pandas_detector = pd.DataFrame()

            print(f'Inter{time_interval}_exp{exposure}_det{detector} -- Starting chunks.')
            for SPIOBSID, files in same_darks.items():
                if len(files) < 3:
                    continue
                data, mads, modes, masks, nb_used, used_images, before_used, after_used = \
                    self.Time_interval(time_interval, exposure, detector, filenames, files, images, positions, SPIOBSID)
                if len(data) == 0:
                    continue

                # Error calculations
                nw_masks, detections, errors, ratio, weights_tot, weights_error, weights_ratio = self.Stats(data, masks,
                                                                                                            modes)

                # # Saving the stats in a csv file
                data_pandas = self.Unique_datadict(time_interval, exposure, detector, files, mads, modes, detections,
                                                   errors, ratio, weights_tot, weights_error, weights_ratio, nb_used)
                csv_name = f'Info_for_ID{SPIOBSID}.csv'
                data_pandas.to_csv(os.path.join(paths['Statistics'], csv_name), index=False)
                data_pandas_detector = pd.concat([data_pandas_detector, data_pandas], ignore_index=True)

                if self.plots:
                    # Plotting the errors
                    self.Error_histo_plotting(paths, nw_masks, data, modes, mads, used_images, before_used, after_used, SPIOBSID, files)

            print(f'Inter{time_interval}_exp{exposure}_det{detector}'
                  f' -- Chunks finished and Median plotting done.', flush=True)

            # Combining the dictionaries
            data_pandas_exposure = pd.concat([data_pandas_exposure, data_pandas_detector], ignore_index=True)
        return data_pandas_exposure

    def Error_histo_plotting(self, paths, error_masks, images, modes, mads, used_images, before_used, after_used, SPIOBSID, files):
        width, rows, cols = np.where(error_masks)


        processed_vals = set()
        for w, r, c in zip(width, rows,  cols):
            if (r, c) in processed_vals:
                continue
            
            processed_vals.add((r, c))
            filename = files[w]
            name_dict = common.SpiceUtils.parse_filename(filename)
            date = parse_date(name_dict['time'])

            before_used_array = before_used[w]
            after_used_array = after_used[w]
            data = np.copy(images[:, r, c])
            data_main = np.copy(used_images[w, :, r, c])
            data_before = np.copy(before_used_array[:, r, c])
            data_after = np.copy(after_used_array[:, r, c])

            bins = self.Bins(data)
            # REF HISTO plotting
            hist_name = f'Error_ID{SPIOBSID}_w{w}_r{r}_c{c}_v2.png'
            plt.hist(data, color='green', bins=bins, label="Same ID data", alpha=0.5)
            if len(data_before) != 0:
                bins = self.Bins(data_before)
                plt.hist(data_before, bins=bins, histtype='step', edgecolor=(0.8, 0.3, 0.3, 0.6))
                plt.hist(data_before, bins=bins, label='Main data before acquisition', color=(0.8, 0.3, 0.3, 0.2))
            if len(data_after) != 0:
                bins = self.Bins(data_after)
                plt.hist(data_after, bins=bins, histtype='step', edgecolor=(0, 0.3, 0.7, 0.6))
                plt.hist(data_after, bins=bins, label='Main data after acquisition', color=(0, 0.3, 0.7, 0.2))
            bins = self.Bins(data[w])
            plt.hist(data[w], bins=bins, label='Studied acquisition', histtype='step', edgecolor='black')
            plt.title(f'Histogram, tot {len(data_main)}, same ID {len(data)}, date {date.year:04d}-{date.month:02d}',
                      fontsize=12)
            plt.xlabel('Detector count', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.axvline(modes[w, r, c], color='magenta', linestyle='--', label='Used mode')
            plt.axvline(modes[w, r, c] + self.coef * mads[w, r, c], color='magenta', linestyle=':',
                        label='Clipping value')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend()
            plt.savefig(os.path.join(paths['Special histograms'], hist_name), bbox_inches='tight', dpi=300)
            plt.close()

    def Time_interval(self, date_interval, exposure, detector, filenames, files, images, positions, SPIOBSID):
        first_filename = files[0]
        name_dict = common.SpiceUtils.parse_filename(first_filename)
        date = parse_date(name_dict['time'])

        year_max = date.year
        year_min = date.year
        month_max = date.month + int(date_interval / 2)
        month_min = date.month - int(date_interval / 2)

        if month_max > 12:
            year_max += (month_max - 1) // 12
            month_max = month_max % 12
        if month_min < 1:
            year_min -= (abs(month_min) // 12) + 1
            month_min = 12 - (abs(month_min) % 12)

        date_max = f'{year_max:04d}{month_max:02d}{date.day:02d}T{date.hour:02d}{date.minute:02d}{date.second:02d}'
        date_min = f'{year_min:04d}{month_min:02d}{date.day:02d}T{date.hour:02d}{date.minute:02d}{date.second:02d}'

        position = []
        for loop, filename in enumerate(filenames):
            name_dict = common.SpiceUtils.parse_filename(filename)
            if (name_dict['time'] >= date_min) and (name_dict['time'] <= date_max):
                position.append(loop)
                if filename == first_filename:
                    first_pos = loop  # global index of the first image with the same ID
                if filename == files[-1]:
                    last_pos = loop  # global index of the last image with the same ID
        position = np.array(position)  # the positions of the files in the right time interval
        timeinit_images = images[position]  # the images in the time interval

        # Making a loop so that the acquisitions with the same ID are not taken into account for the mad and mode
        mads = []
        modes = []
        masks = []
        nb_used_images = []
        used_images = []
        before_used_images = []
        after_used_images = []
        for loop in range(len(files)):
            index_n = first_pos - position[0] + loop  # index of the image in the timeint_images array

            delete1_init = first_pos - position[0]  # first pos in the reference frame of timeinit_images
            delete1_end = index_n  # index of the image in timeinit_images
            delete2_init = index_n + 1
            delete2_end = last_pos + 1 - position[0]

            delete1 = np.arange(delete1_init, delete1_end)
            delete2 = np.arange(delete2_init, delete2_end)
            delete_tot = np.concatenate((delete1, delete2), axis=0)

            delete_before = np.arange(0, delete2_end)
            delete_after = np.arange(delete1_init, len(timeinit_images))
            before_timeinit_images = np.delete(timeinit_images, delete_after, axis=0)
            after_timeinit_images = np.delete(timeinit_images, delete_before, axis=0)
            nw_timeinit_images = np.delete(timeinit_images, delete_tot, axis=0)  # Used images without the same IDs
            nw_length = len(nw_timeinit_images)

            print(f'Inter{date_interval}_exp{exposure}_det{detector}_ID{SPIOBSID}'
                  f' -- Nb of used files: {nw_length}', flush=True)

            if nw_length < self.set_min:
                print(f'\033[31mInter{date_interval}_exp{exposure}_det{detector}_ID{SPIOBSID} '
                      f'-- Less than {self.set_min} files. Going to next SPIOBSID\033[0m')
                return [], [], [], [], [], [], [], []

            mad, mode, mask = self.Mad_mean_mask(nw_timeinit_images)
            image_index = index_n - len(delete1)
            mads.append(mad)
            modes.append(mode)
            masks.append(mask[image_index])
            nb_used_images.append(nw_length)
            used_images.append(nw_timeinit_images)
            before_used_images.append(before_timeinit_images)
            after_used_images.append(after_timeinit_images)
        mads = np.array(mads)
        modes = np.array(modes)
        masks = np.array(masks)  # all the masks for the images with the same ID
        nb_used_images = np.array(nb_used_images)
        used_images = np.array(used_images)

        loops = positions[SPIOBSID]
        data = images[loops]  # all the images with the same ID
        return data, mads, modes, masks, nb_used_images, used_images, before_used_images, after_used_images

    def Unique_datadict(self, time_interval, exposure, detector, files, mads, modes, detections, errors, ratio,
                        weights_tot, weights_error, weights_ratio, nb_used):
        """
        Function to create a dictionary containing some useful information on each exposure times. This is
        done to save the stats in a csv file when the code finishes running.
        """

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
        times, a, b, c, d, e, f, g, h = np.full((group_nb, 9), [time_interval, exposure, detector, group_date, group_nb,
                                                                tot_detection, tot_error, tot_ratio, SPIOBSID]).T

        data_dict = {'Time interval': times, 'Exposure time': a, 'Detector': b, 'Group date': c,
                     'Nb of files with same ID': d, 'Tot nb of detections': e, 'Tot nb of errors': f,
                     'Ratio errors/detections': g, 'Filename': files, 'SPIOBSID': h, 'Average Mode': np.mean(modes),
                     'Average mode absolute deviation': np.mean(mads), 'Nb of used images': nb_used,
                     'Nb of detections': detections, 'Nb of errors': errors, 'Ratio': ratio,
                     'Weighted detections': weights_tot, 'Weighted errors': weights_error,
                     'Weighted ratio': weights_ratio}

        Pandasdata = pd.DataFrame(data_dict)
        return Pandasdata

    def mode_along_axis(self, arr):
        return np.bincount(arr).argmax()

    def Mad_mean_mask(self, images):
        """
        Function to calculate the mad, mode and mask for a given chunk
        (i.e. spatial chunk with all the temporal values).
        """

        # Binning the data
        binned_arr = (images // self.bins) * self.bins

        modes = np.apply_along_axis(self.mode_along_axis, 0, binned_arr).astype('float32')  #a double list comprehension with reshape could be faster
        mads = np.mean(np.abs(images - modes), axis=0).astype('float32')

        # Mad clipping to get the chunk specific mask
        masks = images > self.coef * mads + modes
        return mads, modes, masks  # these are all the values for each chunk

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

    def Stats(self, data, masks, meds):
        """
        To calculate some stats to have an idea of the efficacy of the method. The output is a set of masks
        giving the positions where the method outputted a worst result than the initial image.
        """

        # Initialisation
        nw_data = np.copy(data)
        data_med = np.mean(data, axis=0).astype('float32')
        meds_dif = data - data_med

        # Difference between the end result and the initial one
        nw_data[masks] = meds[masks]
        nw_meds_dif = nw_data - data_med

        # Creating a new set of masks that shows where the method made an error
        nw_masks = np.abs(nw_meds_dif) > np.abs(meds_dif)

        ### MAIN STATS
        # Initialisation of the corresponding matrices
        weights_errors = np.zeros(np.shape(data))
        weights_errors[nw_masks] = np.abs(np.abs(meds_dif[nw_masks]) - np.abs(nw_meds_dif[nw_masks]))
        weights_tots = np.abs(np.abs(meds_dif) - np.abs(nw_meds_dif))
        # Calculating the number of detections and errors per dark
        detections = np.sum(masks, axis=(1, 2))
        errors = np.sum(nw_masks, axis=(1, 2))
        # Calculating the "weighted error"
        weights_error = np.sum(weights_errors, axis=(1, 2))
        weights_tot = np.sum(weights_tots, axis=(1, 2))
        # Calculating the ratios
        if 0 not in detections:  # if statements separates two cases (with or without detections)
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

    def Bins(self, data):
        """
        Small function to calculate the appropriate bin count.
        """

        return np.arange(int(np.min(data)) - self.bins/2, int(np.max(data)) + self.bins, self.bins)


if __name__ == '__main__':
    mpl.rcParams['figure.figsize'] = (8, 8)

    import sys

    print(f'version is {sys.version}', flush=True)

    warnings.filterwarnings('ignore', category=mpl.MatplotlibDeprecationWarning)
    test = Cosmicremoval_class(min_filenb=20)
    test.Multiprocess()
    warnings.filterwarnings("default", category=mpl.MatplotlibDeprecationWarning)

