# IMPORTS
from __future__ import annotations
import os
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from itertools import product
from collections import Counter
from typeguard import typechecked
from multiprocessing import Process, Manager, Pool
from dateutil.parser import parse as parse_date
from multiprocessing.queues import Queue as QUEUE

# Local Python files
from Common import RE, SpiceUtils, Decorators, MultiProcessing


# Filtering initialisation
cat = SpiceUtils.read_spice_uio_catalog()
filters = cat.STUDYDES.str.contains('dark') & (cat['LEVEL'] == 'L1')
init_res = cat[filters]


# Creating a class for the cosmicremoval
class CosmicRemoval:
    """To get the SPICE darks, separate them depending on exposure time, do a temporal analysis of the detector counts for each 
    pixels, flag any values that are too far from the distribution as cosmic rays to then replace them by the mode of the detector 
    count for said pixel.
    """

    # Filename pattern
    filename_pattern = re.compile(
        r"""solo
        _(?P<level>L[123])
        _spice
            (?P<concat>-concat)?
            -(?P<slit>[wn])
            -(?P<type>(ras|sit|exp))
            (?P<db>-db)?
            (?P<int>-int)?
        _(?P<time>\d{8}T\d{6})
        _V(?P<version>\d{2})
        _(?P<SPIOBSID>\d+)-(?P<RASTERNO>\d+)
        \.fits
        """,
        re.VERBOSE)

    @typechecked
    def __init__(self, processes: int = 10, max_date: str | None = '20230402T030000', coefficient: int | float = 6, min_filenb: int = 20, 
                 set_min: int = 4, time_interval: int = 6, bins: int = 5, plots: bool = False, fits: bool = True, statistics: bool = True, verbose: int = 1):
        """To initialise but also run the CosmicRemoval class.

        Args:
            multiprocessing (bool, optional): to use multiprocessing or not. Defaults to True.
            max_date (str | None, optional): the maximum date for which the treatment is done on all darks. Defaults to '20230402T030000'.
            coefficient (int | float, optional): the m.a.d. multiplication coefficient that decides what is flagged as a cosmic collision. Defaults to 6.
            min_filenb (int, optional): the minimum number of files needed in a specific exposure time set to start the treatment. Defaults to 20.
            set_min (int, optional): the minimum number of files need to consider a SPIOBSID set as single darks. Defaults to 6.
            time_interval (int, optional): the time interval considered in months (e.g. 6 is 3 months prior and after each acquisition). Defaults to 6.
            bins (int, optional): the binning value in detector counts for the histograms. Defaults to 5.
            verbose (int, optional): Decides how precise you want your logging prints to be, 0 being not prints and 2 being the maximum. Defaults to 1.
        """
    
        # Arguments
        self.multiprocessing = True if processes > 1 else False
        self.processes = processes
        self.coef = coefficient
        self.min_filenb = min_filenb
        self.set_min = set_min
        self.time_interval = time_interval  # time interval considered in months (e.g. 6 is 3 months prior and after each acquisition)
        self.bins = bins
        self.making_plots = plots
        self.making_statistics = statistics
        self.creating_fits = fits
        self.max_date = max_date if max_date is not None else '30000100T000000'  # maximum date for which the treatment is done for multi-darks
        self.verbose = verbose

        # Code functions
        self.exposures = self.Exposure()
        self.Multiprocess()

    ################################################ INITIAL functions #################################################
    def Paths(self, time_interval: int = -1, exposure: float = -0.1, detector: int = -1) -> dict[str, str]:
        """Function to create all the different paths. Lots of if statements to be able to create the needed directories 
        depending on the arguments.

        Args:
            time_interval (int, optional): if not -1, creates the corresponding time_interval directory. Defaults to -1.
            exposure (float, optional): if not -1, creates the corresponding exposure directory. Defaults to -0.1.
            detector (int, optional): if not -1, creates the corresponding directory. Defaults to -1.
            fits (bool, optional): if True, then cosmic treated FITS files are created. IMPORTANT: The files will have a 
            wrong filename and header info... Defaults to True.

        Returns:
            dict[str, str]: the dictionary with keys as argument names (except the 'main' key) and values as the path to the 
            corresponding directory.
        """

        main_path = os.path.join(os.getcwd(), 'statisticsAndPlots')

        if time_interval != -1:
            time_path = os.path.join(main_path, f'Date_interval{time_interval}')

            if exposure != -1:
                exposure_path = os.path.join(time_path, f'Exposure{exposure}')

                if detector != -1:
                    detector_path = os.path.join(exposure_path, f'Detector{detector}')

                    # Main paths
                    initial_paths = {
                        'main': main_path,
                        'time_interval': time_path, 
                        'exposure': exposure_path,
                        'detector': detector_path,
                    }

                    # Secondary paths
                    directories = []
                    if self.fits: directories.append('FITS')
                    if self.making_statistics: directories.append('Statistics')
                    if self.making_plots: directories.append('Special_histograms')

                    paths = {}
                    for directory in directories:
                        path = os.path.join(detector_path, directory)
                        paths[directory] = path
                else:
                    initial_paths = {
                        'Main': main_path, 
                        'Time interval': time_path, 
                        'Exposure': exposure_path,
                    }
                    paths = {}
            else:
                initial_paths = {
                    'Main': main_path,
                    'Time interval': time_path,
                }
                paths = {}
        else:
            initial_paths = {'Main': main_path}
            paths = {}

        all_paths = {}
        for d in [initial_paths, paths]:
            for _, path in d.items():
                os.makedirs(path, exist_ok=True)
            all_paths.update(d)
        return all_paths
    
    def Exposure(self) -> list[float]:
        """Function to find the different exposure times in the SPICE catalogue.

        Returns:
            list[float]: exposure times that will be processed.
        """

        # Getting the exposure times and nb of occurrences
        exposure_counts = Counter(init_res.XPOSURE)
        exposure_weighted = np.array(list(exposure_counts.items()))

        # Printing the values
        if self.verbose > 0: [print(f'For exposure time {exposure[0]}s there are {int(exposure[1])} darks.') for exposure in exposure_weighted]

        # Keeping the exposure times with enough darks
        occurrences_filter = (exposure_weighted[:, 1] > self.min_filenb)
        exposure_used = exposure_weighted[occurrences_filter][:, 0]

        if self.verbose > 0:
            print(f'\033[93mExposure times with less than \033[1m{self.min_filenb}\033[0m\033[93m darks are not kept.\033[0m')
            print(f'\033[33mExposure times kept are {exposure_used}\033[0m')

        if self.making_statistics:
            # Saving exposure stats
            paths = self.Paths()
            csv_name = 'Nb_of_darks.csv'
            exp_dict = {
                'Exposure time (s)': exposure_weighted[:, 0],
                'Total number of darks': exposure_weighted[:, 1],
            }
            pandas_dict = pd.DataFrame(exp_dict)
            sorted_dict = pandas_dict.sort_values(by='Exposure time (s)')

            total_darks = sorted_dict['Total number of darks'].sum().round()
            total_row = pd.DataFrame({
                'Exposure time (s)': ['Total'], 
                'Total number of darks': [total_darks],
            })
            sorted_dict = pd.concat([sorted_dict, total_row], ignore_index=True)
            sorted_dict.to_csv(os.path.join(paths['Main'], csv_name), index=False)
        return exposure_used
    
    def Images_all(self, exposure: float) -> list[str | None]:
        """Function to get, for a certain exposure time, the corresponding filenames.

        Args:
            exposure (float): the exposure time that is going to be studied.

        Returns:
            list[str | None]: list of all the usable filenames found for the exposure time.
        """

        # Filtering the data by exposure time
        filters = (init_res.XPOSURE == exposure)
        res = init_res[filters]

        # All filenames
        filenames = np.array(list(res['FILENAME']))

        # Variable initialisation
        a = 0
        left_nb = 0
        weird_nb = 0
        all_files = []
        for files in filenames:
            # Opening the files
            header = fits.getheader(SpiceUtils.ias_fullpath(files), 0)

            if header['BLACKLEV'] == 1: left_nb += 1; continue
            OBS_DESC = header['OBS_DESC'].lower()
            if 'glow' in OBS_DESC: weird_nb += 1; continue # Weird "glow" darks

            temp1 = header['T_SW']
            temp2 = header['T_LW']

            if temp1 > 0 and temp2 > 0: a += 1; continue

            all_files.append(files)

        if self.verbose > 1:
            if a != 0: print(f'\033[31mExp{exposure} -- Tot nb files with high temp: {a}\033[0m')
            if left_nb != 0: print(f'\033[31mExp{exposure} -- Tot nb files with bias subtraction: {left_nb}\033[0m')
            if weird_nb != 0: print(f'\033[31mExp{exposure} -- Tot nb of "weird" files: {weird_nb}\033[0m')

        if self.verbose > 0: print(f'Exp{exposure} -- Nb of "usable" files: {len(all_files)}')

        if len(all_files) == 0: 
            if self.verbose > 0: print(f'\033[91m ERROR: NO "USABLE" ACQUISITIONS FOR EXP{exposure}- CHANGING EXPOSURE\033[0m')
            return [None]
        return all_files

    ############################################### CALCULATION functions ##############################################
    @Decorators.running_time
    def Multiprocess(self) -> None:
        """To run multiple processes if multiprocessing=True.
        """

        # Choosing to multiprocess or not
        if self.multiprocessing:
            # Setting up the multiprocessing 
            manager = Manager()
            queue = manager.Queue()
            indexes = MultiProcessing.Pool_indexes(len(self.exposures), self.processes)

            if self.verbose > 0: print(f'Number of processes used is: {len(indexes)}.', flush=True)

            processes = [Process(target=self.Main, args=(queue, index)) for index in indexes]            
            for p in processes: p.start()
            for p in processes: p.join()

            results = []
            while not queue.empty(): results.append(queue.get())
        else:
            results = self.Main((0, len(self.exposures) - 1))

        if self.making_statistics: self.Saving_main_numbers(results)

    def Saving_main_numbers(self, results: tuple[float, int, list[str]]) -> None:
        """Saving the number of files processed and which SPIOBSID were processed.

        Args:
            results (tuple[float, int, list[str]]): contains the exposure time, the nb of files and the corresponding filenames list.
        """

        processed_nb_df = pd.DataFrame([
            (exposure, nb)
            for exposure, nb, _ in results
            ], columns=['Exposure', 'Processed'])
        
        processed_nb_df.sort_values(by='Exposure')
        total_processed = processed_nb_df['Processed'].sum()
        total_processed_df = pd.DataFrame({
            'Exposure': ['Total'],
            'Processed': [total_processed],
        })
        processed_nb_df = pd.concat([processed_nb_df, total_processed_df], ignore_index=True)

        filenames_df = pd.DataFrame([
            filename 
            for _, _, filename_list in results 
            for filename in filename_list
            ], columns=['Filenames'])
        filenames_df.sort_values(by='Filenames')
        
        # Saving both stats
        df1_name = 'Nb_of_processed_darks.csv'
        processed_nb_df.to_csv(df1_name, index=False)
        df2_name = 'Processed_filenames.csv'
        filenames_df.to_csv(df2_name, index=False)
        
    def Main(self, index: tuple[int, int], queue: QUEUE | None = None) -> None | list[tuple[float, int, list[str]]]:
        """Main structure of the code after the multiprocessing. Does the treatment given an exposure time and outputs the results.

        Args:
            queue (QUEUE): a multiprocessing.Manager.Queue() object.
            exposure (float): the exposure time to be analysed.

        Raises:
            ValueError: if the filenames doesn't match the usual SPICE FITS filenames.
            ValueError: if there are no NANs in the acquisitions but the header information said there are.
            ValueError: if there are NANs in the acquisitions but the header information didn't say so.

        Returns:
            None | tuple[float, int, list[str]]: contains the exposure time, the nb of files and the corresponding filenames list if not multiprocessing. Else,
            just populates the queue with those values.
        """        

        if self.verbose > 2: print(f'The process id is {os.getpid()}')

        if not self.multiprocessing: results = [None] * len(self.exposures)
        for i, exposure in enumerate(self.exposures[index[0]:index[1] + 1]):
            filenames = self.Images_all(exposure)
            
            # Initialisation of the stats for csv file saving
            if self.making_statistics: exposure_pandas = pd.DataFrame()
            
            if len(filenames) < self.min_filenb:
                if self.verbose > 0:
                    print(f'\033[91mInter{self.time_interval}_exp{exposure} -- Less than {self.min_filenb} usable files. '
                        f'Changing exposure times.\033[0m')
                return

            # MULTIPLE DARKS analysis
            same_darks = self.Samedarks(filenames)

            processed_darks_total_nb = 0
            processed_filenames = []
            for loop, filename in enumerate(filenames):
                pattern_match = CosmicRemoval.filename_pattern.match(filename)

                if pattern_match:
                    SPIOBSID = pattern_match.group('SPIOBSID')
                    date = pattern_match.group('time')
                    new_filename = RE.replace_group(pattern_match, 'version', f'{99}')
                else:
                    raise ValueError(f"The filename {filename} doesn't match the expected pattern.")
                
                length = len(same_darks[SPIOBSID])
                if length > 3:
                    if date > self.max_date:
                        if self.verbose > 1:
                            print(f'\033[31mInter{self.time_interval}_exp{exposure}_imgnb{loop}'
                                f'-- Image from a SPIOBSID set of {length}. Going to the next acquisition.\033[0m')
                        continue
                    elif self.verbose > 1:
                        print(f'Image from a SPIOBSID set of {length} darks but before May 2023.')
                
                interval_filenames = self.Time_interval(filename, filenames)
                
                if len(interval_filenames) < self.set_min:
                    if self.verbose > 1:
                        print(f'\033[31mInter{self.time_interval}_exp{exposure}_imgnb{loop} '
                            f'-- Less than {self.set_min} files for the processing. Going to the next filename.\033[0m')
                    continue

                # For stats
                processed_darks_total_nb += 1
                processed_filenames.append(filename)                
                
                new_images = []
                check = False
                for detector in range(2):
                    mode, mad = self.Mad_mean(interval_filenames, detector)
                    if os.path.exists(os.path.join(SpiceUtils.ias_fullpath(filename))):
                        image = np.array(fits.getdata(SpiceUtils.ias_fullpath(filename), detector)[0, :, :, 0], dtype='float64')
                    else:
                        if self.verbose > 2: print("filename doesn't exist, adding a +1 to the version number")
                        image = np.array(fits.getdata(SpiceUtils.ias_fullpath(new_filename), detector)[0, :, :, 0], dtype='float64')
                    mask = image > self.coef * mad + mode
                    
                    nw_image = np.copy(image)
                    nw_image[mask] = mode[mask]
                    nw_image = nw_image[np.newaxis, :, :, np.newaxis]
                    if np.isnan(image).any():
                        if self.verbose > 2: print(f'\033[1;94mImage contains nans: {np.isnan(nw_image).any()}\033[0m')
                        nw_image[np.isnan(nw_image)] = 65535 
                        check = True
                    new_images.append(nw_image)

                new_images = np.array(new_images, dtype='uint16')

                # Headers
                init_header_SW = fits.getheader(SpiceUtils.ias_fullpath(filename), 0)
                init_header_LW = fits.getheader(SpiceUtils.ias_fullpath(filename), 1)

                if (init_header_LW['NLOSTPIX'] != 0) or (init_header_SW['NLOSTPIX'] != 0):
                    if not check: raise ValueError(f"There should have been a bright blue print before but didn't happen. Filename: {filename}")
                else:
                    if check: raise ValueError(f'Nans where found even though there are none in the headers. Filename: {filename}')

                # Creating the total hdul
                hdul_new = []
                hdul_new.append(fits.PrimaryHDU(data=new_images[0], header=init_header_SW))
                hdul_new.append(fits.ImageHDU(data=new_images[1], header=init_header_LW))
                hdul_new = fits.HDUList(hdul_new)
                hdul_new.writeto(new_filename, overwrite=True)

                if self.verbose > 0: print(f'File nb{loop}, i.e. {filename}, processed.', flush=True)
            if self.verbose > 0: print(f'\033[1;33mFor exposure {exposure}, {processed_darks_total_nb} files processed\033[0m')

            if self.multiprocessing: 
                queue.put((exposure, processed_darks_total_nb, processed_filenames))
            else:
                results[i] = (exposure, processed_darks_total_nb, processed_filenames)
        if not self.multiprocessing: return results

    def Time_interval(self, filename: str, files: list[str | None]) -> list[str]:
        """Finding the images in the time interval specified for a given filename. Hence, we are getting the sequences of images needed for each individual
        dark treatment.

        Args:
            filename (str): the dark filename to be treated
            files (np.ndarray): list of the useful filenames for that exposure time.

        Returns:
            list[str]: list of the filenames in the given interval time.
        """

        name_dict = SpiceUtils.parse_filename(filename)
        date = parse_date(name_dict['time'])

        year_max = date.year
        year_min = date.year
        month_max = date.month + int(self.time_interval / 2)
        month_min = date.month - int(self.time_interval / 2)

        if month_max > 12:
            year_max += (month_max - 1) // 12
            month_max = month_max % 12
        if month_min < 1:
            year_min -= (abs(month_min) // 12) + 1
            month_min = 12 - (abs(month_min) % 12)

        date_max = f'{year_max:04d}{month_max:02d}{date.day:02d}T{date.hour:02d}{date.minute:02d}{date.second:02d}'
        date_min = f'{year_min:04d}{month_min:02d}{date.day:02d}T{date.hour:02d}{date.minute:02d}{date.second:02d}'

        interval_filenames = []
        for interval_filename in files:
            if interval_filename == filename: continue

            name_dict = SpiceUtils.parse_filename(interval_filename)
            date = name_dict['time']

            if (date >= date_min) and (date <= date_max): interval_filenames.append(interval_filename)
        return interval_filenames

    def mode_along_axis(self, arr: np.ndarray) -> int:
        """Outputs the mode of a given binned array.

        Args:
            arr (np.ndarray): the pre-binned data array.

        Returns:
            int: the mode of the array.
        """

        return np.bincount(arr.astype('int64')).argmax()
    
    def Mad_mean(self, filenames: list[str], detector: int) -> tuple[np.ndarray, np.ndarray]:
        """To calculate the m.a.d. and mode for a given time interval chunk.

        Args:
            filenames (list[str]): list of the filenames in the time chunk.
            detector (int): the detector number (i.e. 0 or 1).

        Returns:
            tuple[np.ndarray, np.ndarray]: gives the mode and mad value for each pixel in the treated dark in question.
        """

        images = np.array([
            np.array(fits.getdata(SpiceUtils.ias_fullpath(filename), detector)[0, :, :, 0], dtype='float64') 
            for filename in filenames
        ]) 

        # Binning the data
        binned_arr = (images // self.bins) * self.bins

        modes = np.apply_along_axis(self.mode_along_axis, 0, binned_arr).astype('float64')  #TODO: a double list comprehension with reshape could be faster
        mads = np.mean(np.abs(images - modes), axis=0).astype('float64')
        return modes, mads 

    def Samedarks(self, filenames: list[str | None]) -> dict[str, str]:
        """To get a dictionary with the keys representing the possible SPIOBSIDs and the values the corresponding filenames.
        Hence, it can also give you the number of darks per SPIOBSID.

        Args:
            filenames (np.ndarray): list of filenames for a given exposure time.

        Returns:
            dict[str, str]: dictionary with .items() = SPIOBSID, corresponding_dark_filenames.
        """

        # Dictionaries initialisation
        same_darks = {}
        for filename in filenames:
            d = SpiceUtils.parse_filename(filename)
            SPIOBSID = d['SPIOBSID']
            if SPIOBSID not in same_darks:
                same_darks[SPIOBSID] = []
            same_darks[SPIOBSID].append(filename)
        return same_darks


class CosmicRemovalStatsPlots:

    @typechecked
    def __init__(self, processes: int = 90, coefficient: int | float = 6, min_filenb: int = 20, set_min: int = 4,
                time_intervals: list[int] = [4, 6, 8], bins: int = 5, statistics: bool = True, plots: bool = True, verbose: int = 0):
        
        # Arguments
        self.processes = processes
        self.coef = coefficient
        self.min_filenb = min_filenb
        self.set_min = set_min
        self.making_statistics = statistics # Bool to know if stats will be saved
        self.making_pots = plots  # boolean to know if plots will be saved
        self.time_intervals = time_intervals
        self.bins = bins
        self.verbose = verbose

        # Code functions
        self.exposures = self.Exposure()

    ################################################ INITIAL functions #################################################
    def Paths(self, time_interval: int = -1, exposure: float = -0.1, detector: int = -1) -> dict[str, str]:
        """
        Function to create all the different paths. Lots of if statements to be able to add files where ever I want
        """
        
        main_path = os.path.join(os.getcwd(), 'statisticsAndPlots')

        if time_interval != -1:
            time_path = os.path.join(main_path, f'Date_interval{time_interval}')

            if exposure != -1:
                exposure_path = os.path.join(time_path, f'Exposure{exposure}')

                if detector != -1:
                    detector_path = os.path.join(exposure_path, f'Detector{detector}')

                    # Main paths
                    initial_paths = {
                        'main': main_path,
                        'time_interval': time_path, 
                        'exposure': exposure_path,
                        'detector': detector_path,
                    }

                    # Secondary paths
                    directories = []
                    if self.making_statistics: directories.append('Statistics')
                    if self.making_plots: directories.append('Special_histograms')

                    paths = {}
                    for directory in directories:
                        path = os.path.join(detector_path, directory)
                        paths[directory] = path
                else:
                    initial_paths = {
                        'Main': main_path, 
                        'Time interval': time_path, 
                        'Exposure': exposure_path,
                    }
                    paths = {}
            else:
                initial_paths = {
                    'Main': main_path,
                    'Time interval': time_path,
                }
                paths = {}
        else:
            initial_paths = {'Main': main_path}
            paths = {}

        all_paths = {}
        for d in [initial_paths, paths]:
            for _, path in d.items(): os.makedirs(path, exist_ok=True)
            all_paths.update(d)
        return all_paths
    
    def Exposure(self) -> list[float]:

        # Getting the exposure times and nb of occurrences
        exposure_counts = Counter(init_res.XPOSURE)
        exposure_weighted = np.array(list(exposure_counts.items()))

        # Printing the values
        if self.verbose > 1: 
            for exposure in exposure_weighted: print(f'For exposure time {exposure[0]}s there are {int(exposure[1])} darks.') 

        # Keeping the exposure times with enough darks
        occurrences_filter = exposure_weighted[:, 1] > self.min_filenb
        exposure_used = exposure_weighted[occurrences_filter][:, 0]
        if self.verbose > 0:
            print(f'\033[93mExposure times with less than \033[1m{self.min_filenb}\033[0m\033[93m darks are not kept.'
                f'\033[0m')
            print(f'\033[33mExposure times kept are {exposure_used}\033[0m')

        if self.making_statistics:
            # Saving exposure stats
            paths = self.Paths()
            csv_name = 'Nb_of_darks.csv'
            exp_dict = {
                'Exposure time (s)': exposure_weighted[:, 0],
                'Total number of darks': exposure_weighted[:, 1],
            }
            pandas_dict = pd.DataFrame(exp_dict)
            sorted_dict = pandas_dict.sort_values(by='Exposure time (s)')

            total_darks = sorted_dict['Total number of darks'].sum().round()
            total_row = pd.DataFrame({
                'Exposure time (s)': ['Total'], 
                'Total number of darks': [total_darks],
            })
            sorted_dict = pd.concat([sorted_dict, total_row], ignore_index=True)
            sorted_dict.to_csv(os.path.join(paths['Main'], csv_name), index=False)
        return exposure_used

    def Images_all(self, exposure: float) -> list[str | None]:
        """
        """

        # Filtering the data by exposure time
        filters = (init_res.XPOSURE == exposure)
        res = init_res[filters]
        filenames = list(res['FILENAME'])

        # Variable initialisation
        a = 0
        left_nb = 0
        weird_nb = 0
        all_files = []
        for files in filenames:
            # Opening the files
            header = fits.getheader(SpiceUtils.ias_fullpath(files), 0)

            if header['BLACKLEV'] == 1: left_nb += 1; continue
            OBS_DESC = header['OBS_DESC'].lower()
            if 'glow' in OBS_DESC: weird_nb += 1; continue # Weird "glow" darks

            temp1 = header['T_SW']
            temp2 = header['T_LW']
            if temp1 > 0 and temp2 > 0: a += 1; continue

            all_files.append(files)

        if self.verbose > 1:
            if a != 0: print(f'\033[31mExp{exposure} -- Tot nb files with high temp: {a}\033[0m')
            if left_nb != 0: print(f'\033[31mExp{exposure} -- Tot nb files with bias subtraction: {left_nb}\033[0m')
            if weird_nb != 0: print(f'\033[31mExp{exposure} -- Tot nb of "weird" files: {weird_nb}\033[0m')

        if self.verbose > 0: print(f'Exp{exposure} -- Nb of "usable" files: {len(all_files)}')

        if len(all_files) == 0: 
            if self.verbose > 0: print(f'\033[91m ERROR: NO "USABLE" ACQUISITIONS FOR EXP{exposure}- CHANGING EXPOSURE\033[0m')
            return [None]
        return all_files

    ############################################### CALCULATION functions ##############################################
    @Decorators.running_time
    def Multiprocess(self):
        """
        Function for multiprocessing if self.processes > 1. No multiprocessing done otherwise.
        """

        # Choosing to multiprocess or not
        if self.processes > 1:
            print(f'Number of used processes is {self.processes}')
            args = list(product(self.time_intervals, self.exposures))
            pool = Pool(processes=self.processes)
            data_pandas_interval = pool.starmap(self.Main, args)
            pool.close()
            pool.join()

            if self.making_statistics: self.Saving_csv(data_pandas_interval)

        else:
            data_pandas_all = pd.DataFrame()

            for time_inter in self.time_intervals:
                data_pandas_interval = pd.DataFrame()

                for exposure in self.exposures:
                    data_pandas_exposure = self.Main(time_inter, exposure)

                    if not self.making_statistics: continue
                    paths = self.Paths(time_interval=time_inter, exposure=exposure)
                    data_pandas_interval = pd.concat([data_pandas_interval, data_pandas_exposure], ignore_index=True)
                    pandas_name0 = f'Alldata_inter{time_inter}_exp{exposure}.csv'
                    data_pandas_exposure.to_csv(os.path.join(paths['Exposure'], pandas_name0), index=False)

                if not self.making_statistics: continue
                data_pandas_all = pd.concat([data_pandas_all, data_pandas_interval], ignore_index=True)
                pandas_name1 = f'Alldata_inter{time_inter}.csv'
                data_pandas_interval.to_csv(os.path.join(paths['Time interval'], pandas_name1), index=False)

            if self.making_statistics: data_pandas_all.to_csv(os.path.join(paths['Main'], 'Alldata.csv'), index=False)

    def Saving_csv(self, data_list):
        args = list(product(self.time_intervals, self.exposures))
        last_time = 0
        first_try = 0

        for loop, pandas_dict in enumerate(data_list):
            indexes = args[loop]
            paths = self.Paths(time_interval=indexes[0], exposure=indexes[1])

            # Saving a csv file for each exposure time
            csv_name = f'Alldata_inter{indexes[0]}_exp{indexes[1]}.csv'
            pandas_dict.to_csv(os.path.join(paths['Exposure'], csv_name), index=False)
            if self.verbose > 1: print(f'Inter{indexes[0]}_exp{indexes[1]} -- CSV files created', flush=True)

            if indexes[0] == last_time:
                pandas_inter = pd.concat([pandas_inter, pandas_dict], ignore_index=True)
                if indexes == args[-1]:
                    pandas_name0 = f'Alldata_inter{indexes[0]}.csv'
                    pandas_inter.to_csv(os.path.join(paths['Time interval'], pandas_name0), index=False)
                    if self.verbose > 0: print(f'Inter{indexes[0]} -- CSV files created', flush=True)
            else:
                if first_try != 0:
                    paths = self.Paths(time_interval=last_time)
                    pandas_name0 = f'Alldata_inter{last_time}.csv'
                    pandas_inter.to_csv(os.path.join(paths['Time interval'], pandas_name0), index=False)
                    if self.verbose > 0: print(f'Inter{indexes[0]} -- CSV files created', flush=True)
                first_try = 1
                last_time = indexes[0]
                pandas_inter = pandas_dict

        data_list = pd.concat(data_list, ignore_index=True)
        pandas_name = 'Alldata.csv'
        data_list.to_csv(os.path.join(paths['Main'], pandas_name), index=False)

    @Decorators.running_time
    def Main(self, time_interval: int, exposure: float):
        if self.verbose> 0: print(f'The process id is {os.getpid()}')
        # Initialisation of the stats for csv file saving
        filenames = self.Images_all(exposure)
        if self.making_statistics: data_pandas_exposure = pd.DataFrame()

        if len(filenames) < self.min_filenb: 
            if self.verbose > 0: print(f'\033[91mInter{time_interval}_exp{exposure} -- Less than {self.min_filenb} usable files. Changing exposure times.\033[0m')
            return

        # MULTIPLE DARKS analysis
        same_darks, positions = self.Samedarks(filenames)
        for detector in range(2):
            if self.making_statistics: data_pandas_detector = pd.DataFrame()
            if self.verbose > 1: print(f'Inter{time_interval}_exp{exposure}_det{detector} -- Starting chunks.')

            paths = self.Paths(time_interval=time_interval, exposure=exposure, detector=detector)

            for SPIOBSID, files in same_darks.items():
                if len(files) < 3: continue
                mads, modes, masks, nb_used, used_filenames, before_used, after_used = \
                    self.Time_interval(time_interval, exposure, detector, filenames, files, SPIOBSID)

                # Error calculations
                nw_masks, detections, errors, ratio, weights_tot, weights_error, weights_ratio = self.Stats(masks, modes, same_darks, SPIOBSID)

                # # Saving the stats in a csv file
                data_pandas = self.Unique_datadict(time_interval, exposure, detector, files, mads, modes, detections,
                                                   errors, ratio, weights_tot, weights_error, weights_ratio, nb_used)
                csv_name = f'Info_for_ID{SPIOBSID}.csv'
                data_pandas.to_csv(os.path.join(paths['Statistics'], csv_name), index=False)
                data_pandas_detector = pd.concat([data_pandas_detector, data_pandas], ignore_index=True)

                if self.making_plots: self.Error_histo_plotting(paths, nw_masks, data, modes, mads, used_images, before_used, after_used, SPIOBSID, files)

            print(f'Inter{time_interval}_exp{exposure}_det{detector}'
                  f' -- Chunks finished and Median plotting done.', flush=True)

            # Combining the dictionaries
            data_pandas_exposure = pd.concat([data_pandas_exposure, data_pandas_detector], ignore_index=True)
        return data_pandas_exposure

    def Error_histo_plotting(self, paths, error_masks, images, modes, mads, used_images, before_used, after_used, SPIOBSID, files):
        width, rows, cols = np.where(error_masks)


        processed_vals = set()
        for w, r, c in zip(width, rows,  cols):
            if (r, c) in processed_vals: continue
            
            processed_vals.add((r, c))
            filename = files[w]
            name_dict = SpiceUtils.parse_filename(filename)
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

    def Time_interval(self, date_interval, exposure, detector, filenames, files, positions, SPIOBSID):
        first_filename = files[0]
        name_dict = SpiceUtils.parse_filename(first_filename)
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
            name_dict = SpiceUtils.parse_filename(filename)
            if (name_dict['time'] >= date_min) and (name_dict['time'] <= date_max):
                position.append(loop)
                if filename == first_filename: first_pos = loop  # global index of the first image with the same ID
                if filename == files[-1]: last_pos = loop  # global index of the last image with the same ID
        position = np.array(position)  # the positions of the files in the right time interval
        # timeinit_images = images[position]  # the images in the time interval
        filenames_init = filenames[position]

        # Making a loop so that the acquisitions with the same ID are not taken into account for the mad and mode
        len_files = len(files)
        mads = [None] * len_files
        modes = [None] * len_files
        masks = [None] * len_files
        nb_used_filenames = [None] * len_files
        used_filenames = [None] * len_files
        before_used_filenames = [None] * len_files
        after_used_filenames = [None] * len_files
        for loop in range(len_files):
            index_n = first_pos - position[0] + loop  # index of the image in the timeint_images array

            delete1_init = first_pos - position[0]  # first pos in the reference frame of timeinit_images
            delete1_end = index_n  # index of the image in timeinit_images
            delete2_init = index_n + 1
            delete2_end = last_pos + 1 - position[0]

            delete1 = np.arange(delete1_init, delete1_end)
            delete2 = np.arange(delete2_init, delete2_end)
            delete_tot = np.concatenate((delete1, delete2), axis=0)

            delete_before = np.arange(0, delete2_end)
            delete_after = np.arange(delete1_init, len(position))
            before_filenames_init = np.delete(filenames_init, delete_after, axis=0)
            after_filenames_init = np.delete(filenames_init, delete_before, axis=0)
            nw_filenames_init = np.delete(filenames_init, delete_tot, axis=0)  # Used images without the same IDs
            nw_length = len(nw_filenames_init)

            if self.verbose > 2: print(f'Inter{date_interval}_exp{exposure}_det{detector}_ID{SPIOBSID} -- Nb of used files: {nw_length}')

            if nw_length < self.set_min:
                if self.verbose > 0: 
                    print(f'\033[31mInter{date_interval}_exp{exposure}_det{detector}_ID{SPIOBSID} -- Less than {self.set_min} files. Going to next SPIOBSID\033[0m')
                return [], [], [], [], [], [], [], []

            mad, mode, mask = self.Mad_mean_mask(nw_filenames_init, detector)
            image_index = index_n - len(delete1)
            mads[loop] = mad
            modes[loop] = mode
            masks[loop] = mask[image_index]
            nb_used_filenames[loop] = nw_length
            used_filenames[loop] = nw_filenames_init
            before_used_filenames[loop] = before_filenames_init
            after_used_filenames[loop] = after_filenames_init
        mads = np.array(mads)
        modes = np.array(modes)
        masks = np.array(masks)  # all the masks for the images with the same ID
        nb_used_filenames = np.array(nb_used_filenames)
        used_filenames = np.array(used_filenames)
        return mads, modes, masks, nb_used_filenames, used_filenames, before_used_filenames, after_used_filenames
    
    def Unique_datadict(self, time_interval, exposure, detector, files, mads, modes, detections, errors, ratio,
                        weights_tot, weights_error, weights_ratio, nb_used):
        """
        Function to create a dictionary containing some useful information on each exposure times. This is
        done to save the stats in a csv file when the code finishes running.
        """

        # Initialisation
        name_dict = SpiceUtils.parse_filename(files[0])
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

    def Mad_mean_mask(self, filenames: list[str], detector: int):
        """
        Function to calculate the mad, mode and mask for a given chunk
        (i.e. spatial chunk with all the temporal values).
        """

        images = np.stack([fits.getdata(filename, detector) for filename in filenames], axis=0)

        # Binning the data
        binned_arr = (images // self.bins) * self.bins

        modes = np.apply_along_axis(self.mode_along_axis, 0, binned_arr).astype('float64')  #a double list comprehension with reshape could be faster
        mads = np.mean(np.abs(images - modes), axis=0).astype('float64')

        # Mad clipping to get the chunk specific mask
        masks = images > self.coef * mads + modes
        return mads, modes, masks  # these are all the values for each chunk

    def Samedarks(self, filenames):
        # Dictionaries initialisation
        same_darks = {}
        positions = {}
        for loop, file in enumerate(filenames):
            d = SpiceUtils.parse_filename(file)
            SPIOBSID = d['SPIOBSID']
            if SPIOBSID not in same_darks:
                same_darks[SPIOBSID] = []
                positions[SPIOBSID] = []
            same_darks[SPIOBSID].append(file)
            positions[SPIOBSID].append(loop)
        return same_darks, positions

    def Stats(self, masks, meds, detector, SPIOBSID, same_darks):
        """
        To calculate some stats to have an idea of the efficacy of the method. The output is a set of masks
        giving the positions where the method outputted a worst result than the initial image.
        """

        data = np.stack([fits.getdata(filename, detector) for filename in same_darks[SPIOBSID]], axis=0)
        
        # Initialisation
        nw_data = np.copy(data)
        data_med = np.median(data, axis=0).astype('float32')
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

    import sys

    print(f'Python version is {sys.version}')

    test = CosmicRemoval(verbose=2)

