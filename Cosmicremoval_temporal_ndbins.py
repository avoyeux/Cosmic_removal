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

    def __init__(self, processes=7, chunk_nb=4, coefficient=6, min_filenb=30, months_interval=12, min_files=12, bins=5):
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
        main_path = os.path.join(os.getcwd(), f'Temporal_coef{self.coef}_{self.months_interval}months_{self.bins}dn_'
                                              f'final')

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
                    temp = hdul[0].header['T_SW']
                else:
                    temp = hdul[0].header['T_LW']
                # if temp > 0:  # TODO: this code won't work if this condition is True for only one detector
                #     a += 1
                #     continue
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
            print('yes1')
            print(f'the cpu count is {os.cpu_count()}')
            print(f'self.processes is {self.processes}')
            processes = []
            args = [(exposure,) for exposure in self.exposures]
            # for loop in range(self.processes):
            #     keyword = {'exposure': args[loop]}
            #     print(f'keyword is :{keyword}')
            #     processes.append(Process(target=self.Main, kwargs=keyword))
            #     processes[-1].start()
            pool = mp.Pool(processes=self.processes)
            pool.starmap(self.Main, args)
            pool.close()
            pool.join()
            print('yes2')
            # for loop in range(self.processes):
            #     processes[loop].join()
        else:
            for exposure in self.exposures:
                self.Main(exposure)

    @decorators.running_time
    def Main(self, exposure):
        print(f'the process id is {os.getpid()}')
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
        print("_______________________")
        for detector in range(2):
            paths = self.Paths(exposure=exposure, detector=detector)

            images, date, dsun = all_images[:, detector], all_date[:, detector], all_dsun[:, detector]
            temp, filenames = all_temp[:, detector], all_filenames[:, detector]

            # # Creation of the ref mad and mode
            # ref_mad, ref_mode = self.Madmode_list_ref(images)

            # # Saving these stats in a csv file
            # new_dict = {'Nb of used files': np.full((2,), len(filenames))}
            # pandas_new = pd.DataFrame(new_dict)
            # csv_name = f'Info_nbfiles{exposure}.csv'
            # pandas_new.to_csv(os.path.join(paths['Main'], csv_name), index=False)

            # # Plotting
            # self.Plotting_init(paths, images, ref_mad, ref_mode)
            # self.Plotting(paths, chunks, chunks_mad, chunks_mode, chunks_mask, date, dsun, temp)
            # self.Plotting_special(paths, chunks, chunks_mask)
            # print(f"Exp{exposure}_det{detector} -- Special and normal histogram plotting done.")

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

            #print(f'Exp{exposure}_det{detector} -- Starting chunks.')
            for SPIOBSID, files in same_darks.items():
                if len(files) < 3:
                    continue
                data, mads, modes, masks = self.time_interval(exposure, detector, filenames, files, images, positions,
                                                            SPIOBSID)
                if len(data) == 0:
                    continue

                # Error calculations
                nw_masks, detections, errors, ratio, weights_tot, weights_error, weights_ratio = self.Stats(data, masks,
                                                                                                            modes)

                # MEDIAN PLOTTING FUNC
                # self.Medianplotting(paths, files, data, masks, mode)
                # self.Stats_plotting(paths, files, data, nw_masks)

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
        year_max = date.year  # TODO: need to add the value to the self.
        year_min = date.year
        month_max = date.month + date_interval
        month_min = date.month - date_interval

        if month_max > 12:
            year_max += (month_max - 1) // 12
            month_max = month_max % 12
        if month_min < 1:
            year_min -= (abs(month_min) // 12) + 1
            month_min += 12 * ((abs(month_min) // 12) + 1)

        date_max = f'{year_max:04d}{month_max:02d}{date.day:02d}T{date.hour:02d}{date.minute:02d}{date.second:02d}'
        date_min = f'{year_min:04d}{month_min:02d}{date.day:02d}T{date.hour:02d}{date.minute:02d}{date.second:02d}'

        position = []
        for loop, file in enumerate(filenames):
            name_dict = common.SpiceUtils.parse_filename(file)
            if (name_dict['time'] >= date_min) and (name_dict['time'] <= date_max):
                position.append(loop)
                if file == first_filename:
                    first_pos = loop
                if file == files[-1]:
                    last_pos = loop
        position = np.array(position)

        # TODO: I am changing the stuff here so that the images with the same ID are not taken into account
        used_images = images[position]
        # mad, mode, chunks_masks = self.Chunks_func(used_images)

        # Making a for loop so that the acquisitions with the same ID are not taken into account for the mad and mode
        mads = []
        modes = []
        masks = []
        paths = self.Paths(exposure, detector)
        for loop in range(len(files)):
            index_n = first_pos - position[0] + loop

            delete1_init = first_pos - position[0]
            delete1_end = index_n
            delete2_init = index_n + 1
            delete2_end = last_pos + 1 - position[0]

            delete1 = np.arange(delete1_init, delete1_end)
            delete2 = np.arange(delete2_init, delete2_end)
            delete_tot = np.concatenate((delete1, delete2), axis=0)

            nw_used_images = np.delete(used_images, delete_tot, axis=0)  # Used images without the same IDs

            print(f'Exp{exposure}_det{detector}_ID{SPIOBSID} -- Nb of used files: {len(nw_used_images)}')

            if len(nw_used_images) < self.min_files:
                print(f'\033[31mExp{exposure}_det{detector}_ID{SPIOBSID} -- Less than {self.min_files} files. '
                      f'Going to next SPIOBSID\033[0m')
                return [], [], [], []

            image_index = index_n - len(delete1)
            mad, mode, chunks_masks = self.Chunks_func(nw_used_images)
            mads.append(mad)
            modes.append(mode)
            masks.append(chunks_masks[image_index])

            # Histo plotting
            self.Plotting_init(paths, nw_used_images, mad, mode, SPIOBSID, loop)

        mads = np.array(mads)
        modes = np.array(modes)
        masks = np.array(masks)

        loops = positions[SPIOBSID]
        data = images[loops]
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
        # modes = np.zeros_like(data)
        # modes[:] = mode
        nw_data[masks] = modes[masks]
        nw_meds_dif = nw_data - data_med

        # Creating a new set of masks that shows where the method made an error
        nw_masks = np.zeros_like(masks, dtype='bool')
        filters = abs(nw_meds_dif) > abs(meds_dif)
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
        if 0 not in detections:  # If statements two separate cases (with or without detections)
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
    def Plotting_init(self, paths, all_images, ref_mad, ref_mode, SPIOBSID, loop):
        """Function to plot the initial histograms"""

        for r in self.ref_pixels:
            for c in self.ref_pixels:
                # Variable initialisation
                data = np.copy(all_images[:, r, c])
                bins = self.Bins(data)

                # REF HISTO plotting
                hist_name = f'histo_{SPIOBSID}_{loop}_r{r}_c{c}.png'
                plt.hist(all_images[:, r, c], bins=bins)
                plt.title(f'mode: {round(ref_mode[r, c], 2)}; mad: {round(ref_mad[r, c], 2)}.', fontsize=12)
                plt.xlabel('Detector count', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.xticks(fontsize=12)
                Cosmicremoval_class.Savefig_config(os.path.join(paths['Histograms'], hist_name))

    def Plotting(self, paths, images, mad, mode, masks, date, dsun, temp):
        """Function to plot most of the needed data"""

        for r in self.ref_pixels:
            for c in self.ref_pixels:
                # Variable initialisation
                data = np.copy(images[:, r, c])
                bins = self.Bins(data)

                # REF HISTO plotting
                hist_name = f'histo_r{r}_c{c}.png'
                plt.hist(data, bins=bins)
                plt.axvline(mode[r, c] - self.coef * mad[r, c], color='red', linestyle='--', label='Clipping value')
                plt.axvline(mode[r, c] + self.coef * mad[r, c], color='red', linestyle='--')
                plt.title(f'mode: {round(mode[r, c], 2)}; mad: {round(mad[r, c], 2)}.')
                plt.xlabel('Detector count')
                plt.ylabel('Frequency')
                plt.xticks(fontsize=12)
                plt.legend(loc=1)
                Cosmicremoval_class.Savefig_config(os.path.join(paths['Histograms'], hist_name))

                # TEMP plotting
                temp_name = f'temp_r{r}_c{c}.png'
                plt.scatter(temp, data)
                plt.title('Detector counts as a function of temperature.')
                plt.xlabel('Temperature [Celsius]')
                plt.ylabel('Counts')
                plt.savefig(os.path.join(paths['Temperatures'], temp_name))
                plt.close()

                # DATE plotting
                date_name = f'date_r{r}_c{c}.png'
                plt.scatter(date, data)
                plt.title('Detector counts as a function of acquisition date.')
                plt.xlabel('Date  [YYYY-MM]')
                plt.ylabel('Counts')
                plt.savefig(os.path.join(paths['Dates'], date_name))
                plt.close()

        # Variable initialisation for total data stats
        med_pixels = np.median(images, axis=(1, 2))
        sum_masks = np.sum(masks, axis=0)

        # TOTAL DATEnDISTANCE plotting
        meddate_name = 'Meddate.png'
        fig, ax1 = plt.subplots()
        ax1.scatter(date, med_pixels, color='b', s=20)
        plt.title('Detector counts and distance to Sun as as a function of acquisition date.')
        ax1.set_xlabel('Date  [YYYY-MM]', fontsize=12)
        ax1.set_ylabel('Median counts', color='b', fontsize=12)
        ax1.tick_params(axis='y', which='major', labelsize=12)
        ax1.tick_params(axis='x', which='major', labelsize=10)
        ax2 = ax1.twinx()
        ax2.scatter(date, dsun, color='g', s=12)
        ax2.set_ylabel('Distance to Sun [m]', color='g', fontsize=12)
        ax2.set_ylim(0.3e11, 2e11)
        ax2.invert_yaxis()
        ax2.tick_params(axis='y', which='major', labelsize=12)
        ax2.tick_params(axis='x', which='major', labelsize=10)
        plt.savefig(os.path.join(paths['Dates'], meddate_name), bbox_inches='tight')
        plt.close()

        # TOTAL temperature plotting
        temp_name = f'medtemp.png'
        plt.scatter(temp, med_pixels)
        plt.title('Median detector counts as a function of temperature.')
        plt.xlabel('Temperature [Celsius]', fontsize=12)
        plt.ylabel('Counts', fontsize=12)
        plt.xticks(fontsize=12)
        Cosmicremoval_class.Savefig_config(os.path.join(paths['Temperatures'], temp_name))

        # SUM MASKS plotting
        plot_name = f'all_masks.pdf'
        plt.imshow(sum_masks, interpolation='none')
        plt.title('Addition of all masks')
        plt.colorbar()
        Cosmicremoval_class.Savefig_config(os.path.join(paths['Masks'], plot_name))

    def Plotting_special(self, paths, chunks, chunks_mask):
        """ Function to plot weird histograms, i.e. pixels that are flagged as cosmics "too" many times"""
        sum_masks = np.sum(chunks_mask, axis=0)

        # To check which pixels are flagged too much
        indexes = np.argwhere(sum_masks > 0.1 * len(chunks_mask))

        # Histograms to look at what's happening
        for index in indexes:
            data = np.copy(chunks[:, index[0], index[1]])
            bins = self.Bins(data)

            hist, bin_edges = np.histogram(data, bins=bins)
            max_bin_index = np.argmax(hist)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mode = bin_centers[max_bin_index]

            # Determination of the mode absolute deviation
            mad = np.mean(np.abs(data - mode))

            # REF HISTO plotting
            histo_name = f'specialhisto_r{index[0]}_c{index[1]}.png'
            plt.hist(data, bins=bins)
            plt.axvline(mode - self.coef * mad, color='red', linestyle='--', label='Clipping value')
            plt.axvline(mode + self.coef * mad, color='red', linestyle='--')
            plt.title(f'mode: {round(mode, 2)}; mad: {round(mad, 2)}.')
            plt.xlabel('Detector count')
            plt.ylabel('Frequency')
            plt.xticks(fontsize=12)
            plt.legend(loc=1)
            Cosmicremoval_class.Savefig_config(os.path.join(paths['Special histograms'], histo_name))

    def Medianplotting(self, paths, files, data, masks, mode):
        """ Function to plot the data for the acquisitions that were done in sets"""
        data_med = np.median(data, axis=0)
        med_dif = data - data_med
        img = np.copy(data)
        modes = np.zeros_like(data)
        modes[:] = mode
        img[masks] = modes[masks]
        med_difnw = img - data_med

        for loop, image in enumerate(data):
            # Initialisation
            file = files[loop]
            mask = masks[loop]
            lines = self.Contours(mask)
            name_dict = common.SpiceUtils.parse_filename(file)
            date = parse_date(name_dict['time'])

            SPIOBSID = name_dict['SPIOBSID']

            # Calculate the first and last percentiles of the image data
            first_percentile = np.percentile(image, 1)
            last_percentile = np.percentile(image, 99.99)
            if first_percentile < 100:
                first_percentile = 100

            # # DIF MEDIAN plotting
            # plot_name = f'Dif_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            # plt.imshow(med_dif[loop], interpolation='none', vmin=-100, vmax=400)
            # plt.title(f"Dif with median: {file}")
            # plt.colorbar()
            # plt.savefig(os.path.join(paths['Medians'], plot_name))
            # # plot_name = f'Dif_cont_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            # # for line in lines:
            # #     plt.plot(line[1], line[0], color='r', linewidth=0.05)
            # # plt.savefig(os.path.join(paths['Medians'], plot_name))
            # plt.close()

            # # DARKS plotting
            # plot_name = f'Dark_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            # plt.imshow(image, interpolation='none')
            # plt.title(f'Dark: {file}')
            # plt.colorbar()
            # plt.savefig(os.path.join(paths['Darks'], plot_name))
            # plot_name = f'Dark_cont_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            # for line in lines:
            #     plt.plot(line[1], line[0], color='r', linewidth=0.05)
            # plt.savefig(os.path.join(paths['Darks'], plot_name))
            # plt.close()
            #
            # # LOG DARKS plotting
            # plot_name = f'Log_dark_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            # fig = plt.imshow(image, interpolation='none')
            # plt.title(f'Dark: {file}')
            # # Create a logarithmic colormap
            # log_cmap = mcolors.LogNorm(vmin=first_percentile, vmax=last_percentile)
            # fig.set_norm(log_cmap)
            # cbar = plt.colorbar(fig)
            # cbar.locator = ticker.MaxNLocator(nbins=5)  # Adjust the number of ticks as desired
            # cbar.update_ticks()
            # plt.savefig(os.path.join(paths['Darks'], plot_name))
            # plot_name = f'Log_darkcont_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            # for line in lines:
            #     plt.plot(line[1], line[0], color='r', linewidth=0.05)
            # plt.savefig(os.path.join(paths['Darks'], plot_name))
            # plt.close()
            #
            # # FINAL plotting
            # plot_name = f'Final_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            # plt.imshow(img[loop], interpolation='none')
            # plt.title(f'Final: {file}')
            # plt.colorbar()
            # plt.savefig(os.path.join(paths['Results'], plot_name))
            # plot_name = f'Final_cont_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            # for line in lines:
            #     plt.plot(line[1], line[0], color='r', linewidth=0.05)
            # plt.savefig(os.path.join(paths['Results'], plot_name))
            # plt.close()
            #
            # # FINAL DIF MEDIAN plotting
            # plot_name = f'Dif_nw_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            # plt.imshow(med_difnw[loop], interpolation='none', vmin=-100, vmax=400)
            # plt.title(f"Dif with median: {file}")
            # plt.colorbar()
            # plt.savefig(os.path.join(paths['Medians'], plot_name))
            # plot_name = f'Dif_nwcont_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            # for line in lines:
            #     plt.plot(line[1], line[0], color='r', linewidth=0.05)
            # plt.savefig(os.path.join(paths['Medians'], plot_name))
            # plt.close()

    def Stats_plotting(self, paths, files, data, nw_masks):
        data_med = np.median(data, axis=0)

        for loop, image in enumerate(data):
            file = files[loop]
            nw_mask = nw_masks[loop]
            lines = self.Contours(nw_mask)
            name_dict = common.SpiceUtils.parse_filename(file)
            date = parse_date(name_dict['time'])
            med_difnw = image - data_med

            # DARK ERROR plotting
            dark_name = f'Dark_errors_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            plt.imshow(image, interpolation='none')
            plt.title(f'Dark: {file}')
            plt.colorbar()
            for line in lines:
                plt.plot(line[1], line[0], color='g', linewidth=0.05)
            Cosmicremoval_class.Savefig_config(os.path.join(paths['Darks'], dark_name))

            # MASK ERROR plotting
            mask_name = f'Mask_errors_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            plt.imshow(nw_mask, interpolation='none')
            plt.title(f'Errors: {file}')
            Cosmicremoval_class.Savefig_config(os.path.join(paths['Masks'], mask_name))

            # Final result plotting with errors
            plot_name = f'Dif_errors_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            plt.imshow(med_difnw, interpolation='none', vmin=-100, vmax=400)
            for line in lines:
                plt.plot(line[1], line[0], color='g', linewidth=0.05)
            plt.colorbar()
            Cosmicremoval_class.Savefig_config(os.path.join(paths['Medians'], plot_name))

    ############################################# MISCELLANEOUS functions ##############################################
    def Bins(self, data):
        """Small function to calculate the appropriate bin count"""
        val_range = np.max(data) - np.min(data)
        # bins = int(len(data) * val_range / 500)  #was 500 before
        bins = np.array(range(int(np.min(data)), int(np.max(data)) + 2, self.bins))
        if len(bins) < 8:
            bins = 8
        return bins

    @staticmethod
    def Contours(mask):  # TODO: change this to a quicker method when all the other stuff is finished
        """Function to plot the contours given a mask
        Source: https://stackoverflow.com/questions/40892203/can-matplotlib-contours-match-pixel-edges"""
        pad = np.pad(mask, [(1, 1), (1, 1)])  # zero padding
        im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
        im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]
        lines = []
        for ii, jj in np.ndindex(im0.shape):
            if im0[ii, jj] == 1:
                lines += [([ii - .5, ii - .5], [jj - .5, jj + .5])]
            if im1[ii, jj] == 1:
                lines += [([ii - .5, ii + .5], [jj - .5, jj - .5])]
        return lines

    @staticmethod
    def Savefig_config(figname, **kwargs):
        savefig_config = {'dpi': 1024 / 8, 'bbox_inches': 'tight'}
        plt.yticks(fontsize=12)
        plt.savefig(figname, **savefig_config, **kwargs)
        plt.close()

if __name__ == '__main__':
    mpl.rcParams['figure.figsize'] = (8, 8)

    warnings.filterwarnings('ignore', category=mpl.MatplotlibDeprecationWarning)
    test = Cosmicremoval_class(min_filenb=30)
    test.Multiprocess()
    warnings.filterwarnings("default", category=mpl.MatplotlibDeprecationWarning)

