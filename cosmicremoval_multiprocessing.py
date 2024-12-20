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
from multiprocessing import Process, Manager
from dateutil.parser import parse as parse_date
from multiprocessing.queues import Queue as QUEUE

# Local Python files
from Common import RE, SpiceUtils, Decorators, Pandas

# Filtering initialisation
cat = SpiceUtils.read_spice_uio_catalog()
filters = cat.STUDYDES.str.contains('dark') & (cat['LEVEL'] == 'L1')
init_res = cat[filters]


class ParentFunctions:
    """Parent class to the Cosmic removal classes as I have created two really similar classes. The first for actually treating the single darks and the
    second to plot and compute some statistics on the multi-darks (to have an idea of the efficiency of the cosmic removal).
    Therefore, this class is only here to store the common functions to both the aforementioned classes.
    """

    def __init__(self):
        """The initialisation of the class.

        Args:
            fits (bool): if True, then cosmic treated FITS files are created. IMPORTANT: The files will have a wrong filename and header info...
            statistics (bool): if True, then statistics are saved in .csv files. 
            plots (bool): if True, then histograms of the errors are saved as .png files. Only works for the non single-darks.
            verbose (int): dictates the precision of the outputted prints related to code running time. The higher the number (3 being the max), the higher
            the number of outputted prints.
        """
        
        # Shared attributes
        self.fits: bool
        self.making_statistics: bool
        self.making_plots: bool
        self.verbose: int

    def paths(self, time_interval: int = -1, exposure: float = -0.1, detector: int = -1) -> dict[str, str]:
        """Function to create all the different paths. Lots of if statements to be able to create the needed directories 
        depending on the arguments.

        Args:
            time_interval (int, optional): if not -1, creates the corresponding time_interval directory. Defaults to -1.
            exposure (float, optional): if not -1, creates the corresponding exposure directory. Defaults to -0.1.
            detector (int, optional): if not -1, creates the corresponding directory. Defaults to -1.

        Returns:
            dict[str, str]: the dictionary with keys representing the directory 'level' (e.g. 'main', 'exposure', 'detector') and values as the path to the 
            corresponding directory.
        """

        initial_paths = {'main': os.path.join(os.getcwd(), 'statisticsAndPlots')}

        if time_interval != -1:
            initial_paths['time_interval'] = os.path.join(initial_paths['main'], f'{time_interval}months')

            if self.fits: initial_paths['fits'] = os.path.join(initial_paths['time_interval'], 'fits')

            if exposure != -0.1:
                initial_paths['exposure'] = os.path.join(initial_paths['time_interval'], f'Exposure{exposure}')

                if detector != -1:
                    initial_paths['detector'] = os.path.join(initial_paths['exposure'], f'Detector{detector}')

                    if self.making_statistics: initial_paths['statistics'] = os.path.join(initial_paths['detector'], 'Statistics')
                    if self.making_plots: initial_paths['histograms'] = os.path.join(initial_paths['detector'], 'Histograms')

        for path in initial_paths.values(): os.makedirs(path, exist_ok=True)
        return initial_paths
    
    @Decorators.running_time
    def main(self, arg_list: list[tuple[int, float, str, str | list[str], list[str]]]) -> None:
        """To run multiple processes if multiprocessing=True.
        """

        # Choosing to multiprocess or not
        if self.multiprocessing:
            # Setting up the multiprocessing 
            manager = Manager()
            input_queue = manager.Queue()
            output_queue = manager.Queue()

            # Choosing the number of processes
            processes_nb = min(self.processes, len(arg_list))
            if self.verbose > 0: print(f'number of processed used are {processes_nb}')

            for arg in arg_list: input_queue.put(arg)
            for _ in range(processes_nb): input_queue.put(None)

            processes = [None] * processes_nb
            for i in range(processes_nb):
                p = Process(target=self.processing, args=(input_queue, output_queue))
                p.start()
                processes[i] = p
            for p in processes: p.join()

            results = []
            while not output_queue.empty(): results.append(output_queue.get())

        else:
            results = []
            for arg in arg_list: 
                result = self.processing(arguments=arg)
                if result is not None:
                    if isinstance(result, list):
                        results.extend(result)
                    else:
                        results.append(result)
        if self.making_statistics: self.saving_main_numbers(results) 

    def exposure(self):
        """Function to find the different exposure times in the SPICE catalogue.

        Returns:
            list[float]: exposure times that will be processed.
        """

        # Getting the exposure times and nb of occurrences
        exposure_counts = Counter(init_res.XPOSURE)
        exposure_weighted = np.stack(list(exposure_counts.items()), axis=0)
        self.exposures = exposure_weighted[:, 0]

        # Printing the values
        if self.verbose > 0: 
            for exposure in exposure_weighted: print(f'For exposure time {exposure[0]}s there are {int(exposure[1])} darks.') 
            print('\033[33mExposure times kept are:')
            for exposure in self.exposures: print(f'\033[33m{exposure}\033[0m')

        if self.making_statistics:
            # Saving exposure stats
            paths = self.paths()
            csv_name = 'Total_darks_info.csv'
            exp_dict = {
                'Exposure time (s)': exposure_weighted[:, 0],
                'Total number of darks': exposure_weighted[:, 1],
            }
            pandas_dict = pd.DataFrame(exp_dict)
            sorted_dict = pandas_dict.sort_values(by='Exposure time (s)')
            sorted_dict['Exposure time (s)'] = sorted_dict['Exposure time (s)']
            total_darks = sorted_dict['Total number of darks'].sum()
            total_row = pd.DataFrame({
                'Exposure time (s)': ['Total'], 
                'Total number of darks': [total_darks],
            })
            sorted_dict = pd.concat([sorted_dict, total_row], ignore_index=True)
            sorted_dict['Total number of darks'] = sorted_dict['Total number of darks'].astype('int')
            sorted_dict.to_csv(os.path.join(paths['main'], csv_name), index=False)
    
    def images_all(self, exposure: float) -> list[str]:
        """Function to get, for a certain exposure time, the corresponding filenames.

        Args:
            exposure (float): the exposure time that is going to be studied.

        Returns:
            list[str | None]: list of all the usable filenames found for the exposure time.
        """

        # Filtering the data by exposure time
        filters = (init_res.XPOSURE == exposure)
        res = init_res[filters]
        filenames = list(res['FILENAME'])

        # Variable initialisation
        a = 0
        left_nb = 0
        weird_nb = 0
        high_voltage = 0
        all_files = []
        for filename in filenames:
            # Opening the files
            header = fits.getheader(SpiceUtils.ias_fullpath(filename), 0)

            # Filtering the unwanted files
            if header['BLACKLEV'] == 1: left_nb += 1; continue
            if 'glow' in header['OBS_DESC'].lower(): weird_nb += 1; continue # Weird "glow" darks
            if header['T_SW'] > 0 and header['T_LW'] > 0: a += 1; continue
            if header['V_GAPSW'] > 2000: high_voltage += 1; continue

            # Keeping the 'good' filenames
            all_files.append(filename)

        if self.verbose > 1:
            if a != 0: print(f'\033[31mExp{exposure} -- Tot nb files with high temp: {a}\033[0m')
            if left_nb != 0: print(f'\033[31mExp{exposure} -- Tot nb files with bias subtraction: {left_nb}\033[0m')
            if weird_nb != 0: print(f'\033[31mExp{exposure} -- Tot nb of "weird" files: {weird_nb}\033[0m')
            if high_voltage != 0: print(f'\033[31mExp{exposure} -- Tot nb of "high voltage" files: {high_voltage}\033[0m')

        if self.verbose > 0: print(f'Exp{exposure} -- Nb of "usable" files: {len(all_files)}')
        if len(all_files) == 0 and self.verbose > 0: print(f'\033[91m ERROR: NO "USABLE" ACQUISITIONS FOR EXP{exposure}\033[0m')
        return all_files
    
    def mode_along_axis(self, arr: np.ndarray) -> int:
        """Outputs the mode of a given binned array.

        Args:
            arr (np.ndarray): the pre-binned data array.

        Returns:
            int: the mode of the array.
        """

        return np.bincount(arr.astype('int32')).argmax()

    def samedarks(self, filenames: list[str]) -> dict[str, str | list[str]]:
        """To get a dictionary with the keys representing the possible SPIOBSIDs and the values the corresponding filenames.
        Hence, it can also give you the number of darks per SPIOBSID.

        Args:
            filenames (np.ndarray): list of filenames for a given exposure time.

        Returns:
            dict[str, str]: dictionary with .items() = (SPIOBSID, corresponding_dark_filenames).
        """

        # Dictionaries initialisation
        same_darks = {}
        for filename in filenames:
            d = SpiceUtils.parse_filename(filename)
            SPIOBSID = d['SPIOBSID']
            if SPIOBSID not in same_darks: same_darks[SPIOBSID] = []  # TODO: need to check what is the difference between .keys() and nothing.
            same_darks[SPIOBSID].append(filename)
        return same_darks
    
    def mad_mode(self, filenames: list[str], detector: int) -> tuple[np.ndarray, np.ndarray]:
        """To calculate the m.a.d. and mode for a given time interval chunk.

        Args:
            filenames (list[str]): list of the filenames in the time chunk.
            detector (int): the detector number (i.e. 0 or 1).

        Returns:
            tuple[np.ndarray, np.ndarray]: gives the mode and mad value for each pixel in the treated dark in question.
        """

        images = np.stack([
            fits.getdata(SpiceUtils.ias_fullpath(filename), detector)[0, :, :, 0]
            for filename in filenames
        ], axis=0) 

        # Binning the data
        binned_arr = (images // self.bins) 

        modes = np.apply_along_axis(self.mode_along_axis, 0, binned_arr).astype('float64') * self.bins #TODO: a double list comprehension with reshape could be faster
        mads = np.mean(np.abs(images - modes), axis=0)
        return mads, modes


class CosmicRemoval(ParentFunctions):
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
    def __init__(self, processes: int = 128, max_date: str | None = '20230402T030000', coefficient: int | float = 6, 
                 set_min: int = 4, time_intervals: int | list[int] = 6, statistics: bool = False, bins: int = 5, verbose: int = 1, flush: bool = False):
        """To initialise but also run the CosmicRemoval class.

        Args:
            processes (int, optional): sets the maximum number of processes for the multi-processing. Defaults to 10.
            max_date (str | None, optional): the maximum date for which the treatment is done on all darks. Defaults to '20230402T030000'.
            coefficient (int | float, optional): the m.a.d. multiplication coefficient that decides what is flagged as a cosmic collision. Defaults to 6.
            set_min (int, optional): the minimum number of files need to consider a SPIOBSID set as single darks. Defaults to 6.
            time_interval (int, optional): the time interval considered in months (e.g. 6 is 3 months prior and after each acquisition). Defaults to 6.
            statistics (bool, optional): if True, then statistics are saved in .csv files. Defaults to False.
            bins (int, optional): the binning value in detector counts for the histograms. Defaults to 5.
            verbose (int, optional): decides how precise you want your logging prints to be, 0 being not prints and 2 being the maximum. Defaults to 1.
            flush (bool, optional): sets the flush argument of some of the main print statements. Defaults to False.
        """
    
        super().__init__()
        # Arguments
        self.fits = False
        self.making_statistics = False
        self.making_plots = False
        
        self.processes = processes
        self.max_date = max_date if max_date is not None else '30000100T000000' 
        self.coef = coefficient
        self.set_min = set_min
        self.time_intervals = time_intervals if isinstance(time_intervals, list) else [time_intervals] 
        self.making_statistics = statistics  # resetting the value as it is not exactly the same in the Parent Class.
        self.bins = bins
        self.verbose = verbose
        self.flush = flush

        # New class attributes
        self.multiprocessing = True if processes > 1 else False

        # Code functions
        self.exposure()
        # Changing the self.exposures values to only contain the value for which I have error statistics
        self.exposures = [0.1, 4.6, 9.6, 14.3, 19.6, 29.6, 59.6, 89.6, 119.6]

        # Argument initialisation 
        exposures_w_filenames = [(exposure, self.images_all(exposure)) for exposure in self.exposures]
        arg_list = [
            (time_inter, exposure, SPIOBSID, SPIOBSID_files, filenames)
            for exposure, filenames in exposures_w_filenames
            if filenames != []
            for SPIOBSID, SPIOBSID_files in self.samedarks(filenames).items()
            for time_inter in self.time_intervals
        ]
        self.main(arg_list)
    
    @Decorators.running_time
    def processing(self, input_queue: QUEUE | None = None, output_queue: QUEUE | None = None, 
                   arguments: tuple[int, float, str, str | list[str], list[str]] | None = None) -> list[tuple[int, float, str]] | None:
        """Main structure of the code after the multiprocessing. Does the treatment given an exposure time and outputs the results.

        Args:
            index (tuple[int, int]): the data's start and end index for each process. 
            queue (QUEUE | None): a multiprocessing.Manager.Queue() object. Defaults to None.

        Returns:
            None | tuple[float, list[str]]: contains the exposure time and the corresponding filenames list if not multiprocessing. Else,
            just populates the queue with those values.
        """        

        if self.verbose > 2: print(f'The process id is {os.getpid()}')

        while True:
            if self.multiprocessing:
                arguments = input_queue.get()
                if arguments is None: return 

            time_inter, exposure, SPIOBSID, SPIOBSID_filenames, filenames = arguments

            # INITIAL DATA CHECKS [...]
            first_filename = SPIOBSID_filenames[0]
            pattern_match = CosmicRemoval.filename_pattern.match(first_filename)
            ## [...] pattern
            if pattern_match: 
                date = pattern_match.group('time')
                new_filename = RE.replace_group(pattern_match, 'version', f'{99}')
            else:
                raise ValueError(f"The filenames for SPIOBSID {SPIOBSID} doesn't match the expected pattern. The filenames are {SPIOBSID_filenames}")
            ## [...] multidarks 
            length = len(SPIOBSID_filenames)
            if length > 3:
                if date > self.max_date:
                    if self.verbose > 1: print(f'\033[31mExp{exposure}_ID{SPIOBSID} -- {len(length)} images in SPIOBSID. Going to the nex ID.\033[0m')
                    if not self.multiprocessing: return 
                    continue
                if self.verbose > 1: print(f'Images from a SPIOBSID set of {length} darks but before May 2023.')
            ## [...] available nb of images
            first_interval_filenames = self.time_integration(time_inter, first_filename, filenames)
            if len(first_interval_filenames) < self.set_min:
                if self.verbose > 1: print(f'\033[31mExp{exposure}_ID{SPIOBSID} -- Set has only {self.set_min} files. Going to the next SPIOBSID.\033[0m')
                if not self.multiprocessing: return 
                continue

            # Creating the corresponding treated fits file
            paths = self.paths(time_interval=time_inter)
            self.fits_creation(paths, first_filename, new_filename, first_interval_filenames)

            # Saving the processed filenames
            result = (time_inter, exposure, first_filename)
            if self.multiprocessing: 
                output_queue.put(result)
            else:
                results = [result]

            # Doing the same for the other filenames with the same SPIOBSID
            if isinstance(SPIOBSID_filenames, list):
                for SPIOBSID_filename in SPIOBSID_filenames[1:]:
                    pattern_match = CosmicRemoval.filename_pattern.match(SPIOBSID_filename)
                    new_filename = RE.replace_group(pattern_match, 'version', f'{99}')

                    # Fits creation
                    interval_filenames = self.time_integration(time_inter, SPIOBSID_filename, filenames)
                    self.fits_creation(paths, SPIOBSID_filename, new_filename, interval_filenames)

                    # Saving results
                    result = (time_inter, exposure, SPIOBSID_filename)
                    if self.multiprocessing: 
                        output_queue.put(result)
                    else:
                        results.append(result)

            if self.verbose > 0: print(f'\033[1;33mInter{time_inter}_exp{exposure}_ID{SPIOBSID}, files processed\033[0m', flush=self.flush)
            if not self.multiprocessing: return results       
            
    def fits_creation(self, paths: dict[str, str], filename: str, new_filename: str, interval_filenames: list[str]):

        treated_images = []
        for detector in range(2):
            image = fits.getdata(SpiceUtils.ias_fullpath(filename), detector)[0, :, :, 0].astype('float64')
            mad, mode = self.mad_mode(interval_filenames, detector)
            mask = image > self.coef * mad + mode

            nw_image = np.copy(image)  # TODO: need to check why np.copy is not the preferred copy method.
            nw_image[mask] = mode[mask]
            nw_image = nw_image[np.newaxis, :, :, np.newaxis]
            treated_images.append(nw_image)

        # Headers
        init_header_SW = fits.getheader(SpiceUtils.ias_fullpath(filename), 0)
        init_header_LW = fits.getheader(SpiceUtils.ias_fullpath(filename), 1)

        # Creating the new hdul
        hdul_new = []
        hdul_new.append(fits.PrimaryHDU(data=treated_images[0].astype('uint16'), header=init_header_SW))
        hdul_new.append(fits.ImageHDU(data=treated_images[1].astype('uint16'), header=init_header_LW))
        hdul_new = fits.HDUList(hdul_new)
        hdul_new.writeto(os.path.join(paths['fits'], new_filename), overwrite=True)

    def saving_main_numbers(self, results: list[tuple[int, float, str] | None]) -> None:
        """Saving the filenames and number of files processed in .csv files.

        Args:
            results (list[tuple[float, list[str]]]): the results gotten from running the code as tuples 
            showing the exposure time value and the corresponding list of processed filenames.
        """

        paths = self.paths()

        # Setting up the data for easier filtering and ordering
        data_df = pd.DataFrame(results, columns=['Time interval [months]', 'Exposure [s]', 'Filename'])
        data_df = data_df.sort_values(by=['Time interval [months]', 'Exposure [s]', 'Filename'])

        # Saving all processed filenames
        data_df_filename = 'Processed_filenames.csv'
        data_df.to_csv(os.path.join(paths['main'], data_df_filename), index=False)

        # Saving the summary
        grouped_data_filename = 'Processed_filenb.csv'
        grouped_data_df = data_df.groupby(['Time interval [months]', 'Exposure [s]']).agg(file_nb=('Filename', 'count')).reset_index()
        grouped_data_df.to_csv(os.path.join(paths['main'], grouped_data_filename), index=False)

        # Stats for each time interval
        for time_interval in data_df['Time interval [months]'].unique():
            paths = self.paths(time_interval=time_interval)
            time_interval_df = data_df[data_df['Time interval [months]'] == time_interval]
            
            # Saving the filenames
            time_interval_filename = f'Processed_filenames_{time_interval}months.csv'
            time_interval_df.to_csv(os.path.join(paths['time_interval'], time_interval_filename), index=False)

            # Saving the summary
            grouped_time_interval_filename = f'Processed_filenb_{time_interval}months.csv'
            grouped_time_interval_df = time_interval_df.groupby(['Time interval [months]', 'Exposure [s]']).agg(file_nb=('Filename', 'count')).reset_index()
            grouped_time_interval_df.to_csv(os.path.join(paths['time_interval'], grouped_time_interval_filename), index=False)

            for exposure in time_interval_df['Exposure [s]'].unique():
                paths = self.paths(time_interval=time_interval, exposure=exposure)
                exposure_df = time_interval_df[time_interval_df['Exposure [s]'] == exposure]

                # Saving filenames
                exposure_filename = f'Processed_filenames_{time_interval}months_{exposure}exposure.csv'
                exposure_df.to_csv(os.path.join(paths['exposure'], exposure_filename), index=False)

                # Saving the summary
                grouped_exposure_filename = f'Processed_filenb_{time_interval}months_{exposure}exp.csv'
                grouped_exposure_df = exposure_df.groupby(['Time interval [months]', 'Exposure [s]']).agg(file_nb=('Filename', 'count')).reset_index()
                grouped_exposure_df.to_csv(os.path.join(paths['exposure'], grouped_exposure_filename), index=False)
        
    def time_integration(self, time_interval: int, filename: str, files: list[str | None]) -> list[str]:
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
        month_max = date.month + int(time_interval / 2)
        month_min = date.month - int(time_interval / 2)

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


class CosmicRemovalStatsPlots(ParentFunctions):
    """To get a visual and statistical idea of the performance of the method. Here, only darks that were taken as a group (i.e. multiple darks with same SPIOBSID) are taken into account to try and flag any
    false detections. The data used to treat each file doesn't include the files that also have the same SPIOBSID (to try and minimize bias). 

    Args:
        ParentFunctions (_type_): To store functions that can be used in this and the CosmicRemoval class.

    Returns: saves some error histogram images and csv files related to the error prediction.
    """

    @typechecked
    def __init__(self, processes: int = 121, coefficient: int | float = 6, set_min: int = 4, time_intervals: list[int] = [6], bins: int = 5, statistics: bool = True, plots: bool = True, 
                plot_ratio: float = 0.01, verbose: int = 0, flush: bool = False):
        """Initialisation of the CosmicRemovalStatsPlots class.

        Args:
            processes (int, optional): _description_. Defaults to 128.
            coefficient (int | float, optional): _description_. Defaults to 6.
            set_min (int, optional): _description_. Defaults to 4.
            time_intervals (list[int], optional): _description_. Defaults to [6].
            bins (int, optional): _description_. Defaults to 5.
            statistics (bool, optional): _description_. Defaults to True.
            plots (bool, optional): _description_. Defaults to True.
            verbose (int, optional): _description_. Defaults to 0.
            flush (bool, optional): _description_. Defaults to False.
        """
        
        super().__init__()
        # Arguments
        self.fits = False
        self.making_statistics = statistics
        self.making_plots = plots
        self.verbose = verbose
        self.processes = processes
        self.coef = coefficient
        self.set_min = set_min
        self.time_intervals = time_intervals
        self.bins = bins
        self.plot_ratio = plot_ratio
        self.flush = flush

        # New class attributes
        self.multiprocessing = True if processes > 1 else False

        # Code functions
        self.exposure()

        # Argument initialisation 
        exposures_w_filenames = [(exposure, self.images_all(exposure)) for exposure in self.exposures]
        arg_list = [
            (time_inter, exposure, SPIOBSID, SPIOBSID_files, filenames)
            for exposure, filenames in exposures_w_filenames
            if filenames != []
            for SPIOBSID, SPIOBSID_files in self.samedarks(filenames).items()
            if len(SPIOBSID_files) >= 3
            for time_inter in self.time_intervals
        ]
        self.main(arg_list)

    @Decorators.running_time
    def processing(self, input_queue: QUEUE | None = None, output_queue: QUEUE | None = None, arguments: None = None) -> pd.DataFrame | None:

        while True:
            if self.multiprocessing:
                # Fetching the inputs
                item = input_queue.get()
                if item is None: return
                time_interval, exposure, SPIOBSID, SPIOBSID_files, filenames = item
            else:
                time_interval, exposure, SPIOBSID, SPIOBSID_files, filenames = arguments

            # Getting the filenames for each time integration
            kwargs = {
                'time_interval': time_interval,
                'filenames': filenames,
                'SPIOBSID_files': SPIOBSID_files,
            }
            used_filenames = self.time_integration(**kwargs)

            # Checking the data length
            total_length = sum(map(len, used_filenames))
            if total_length < self.set_min:  
                if self.verbose > 0: print(f'\033[31mInter{time_interval}_exp{exposure}_ID{SPIOBSID} -- Less than {self.set_min} files. Going to next SPIOBSID\033[0m')
                if self.multiprocessing: continue
                return
        
            for detector in range(2):
                if self.making_statistics: data_pandas_detector = pd.DataFrame()

                # Path initialisation
                paths = self.paths(time_interval=time_interval, exposure=exposure, detector=detector)

                # Getting the treatment data for each same SPIOBSID file
                kwargs = {
                    'detector': detector,
                    'SPIOBSID_files': SPIOBSID_files,
                    'filenames_interval': used_filenames,
                }
                data = self.data_bundle(**kwargs)

                # Error calculations
                nw_masks, detections, errors, ratio, weights_tot, weights_error, weights_ratio = self.stats(data)

                # Error plotting
                kwargs = {
                    'data': data,
                    'paths': paths,
                    'error_masks': nw_masks,
                    'SPIOBSID_files': SPIOBSID_files, 
                    'filenames_interval': used_filenames,
                    'SPIOBSID': SPIOBSID,
                    'detector': detector, 
                }
                if self.making_plots: self.error_histo_plotting(**kwargs)
                if self.verbose > 0: print(f'int{time_interval}_exp{exposure}_det{detector}_ID{SPIOBSID} -- Histograms done.', flush=self.flush)

                # Saving the stats
                if not self.making_statistics: continue
                kwargs = {
                    'time_interval': time_interval,
                    'exposure': exposure,
                    'detector': detector,
                    'SPIOBSID_files': SPIOBSID_files,
                    'detections': detections,
                    'errors': errors,
                    'ratio': ratio,
                    'weights_tot': weights_tot,
                    'weights_error': weights_error,
                    'weights_ratio': weights_ratio,
                    'filenames_interval': used_filenames,
                }
                data_pandas = self.unique_datadict(**kwargs)
                csv_name = f'Info_for_ID{SPIOBSID}.csv'
                data_pandas.to_csv(os.path.join(paths['statistics'], csv_name), index=False)
                data_pandas_detector = pd.concat([data_pandas_detector, data_pandas], ignore_index=True)

            if self.verbose > 1: print(f'Inter{time_interval}_exp{exposure}_ID{SPIOBSID} -- Chunks finished and histogram plotting done.', flush=self.flush)
   
            if self.making_statistics:
                if not self.multiprocessing: return data_pandas_detector
                output_queue.put(data_pandas_detector) 
            elif not self.multiprocessing:
                return                 

    def saving_main_numbers(self, data_list: list[pd.DataFrame]) -> None:

        data_df = pd.concat(data_list, ignore_index=True)
        data_df = data_df.sort_values(by=['Time interval [months]', 'Exposure time [s]', 'Group date', 'SPIOBSID'])

        # Saving all the data
        paths = self.paths()
        data_df_filename = 'All_errors.csv'
        data_df.to_csv(os.path.join(paths['main'], data_df_filename), index=False)

        # For each time interval
        for time_interval in data_df['Time interval [months]'].unique():
            paths = self.paths(time_interval=time_interval)
            month_df = data_df[data_df['Time interval [months]'] == time_interval]

            # Saving for each time interval
            month_df_filename = f'Errors_{time_interval}months.csv'
            month_df.to_csv(os.path.join(paths['time_interval'], month_df_filename), index=False)

            for exposure in month_df['Exposure time [s]'].unique():
                paths = self.paths(time_interval=time_interval, exposure=exposure)
                exposure_df = month_df[month_df['Exposure time [s]'] == exposure]

                # Saving for each exposure time 
                exposure_df_filename = f'Errors_{time_interval}months_{exposure}exp.csv'
                exposure_df.to_csv(os.path.join(paths['exposure'], exposure_df_filename), index=False)
        if self.verbose > 0: print('Main statistics saved in csv file. Creating the corresponding average statistics')
        self.average_statistics()

    def data_bundle(self, detector: int, SPIOBSID_files: list[str], filenames_interval: tuple[list[str], list[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

         # Initialisation
        before_filenames, after_filenames = filenames_interval
        all_filenames = before_filenames + after_filenames

        files_len = len(SPIOBSID_files)
        images_SPIOBSID = [None] * files_len
        mads = [None] * files_len
        modes = [None] * files_len
        masks = [None] * files_len

        for i, main_filename in enumerate(SPIOBSID_files):
            image = fits.getdata(SpiceUtils.ias_fullpath(main_filename), detector)[0, :, :, 0].astype('float64')
            data_used = all_filenames + [main_filename]
            mad, mode = self.mad_mode(data_used, detector)
            mask = image > self.coef * mad + mode

            images_SPIOBSID[i] = image
            mads[i] = mad
            modes[i] = mode
            masks[i] = mask

        images_SPIOBSID = np.stack(images_SPIOBSID, axis=0)
        mads = np.stack(mads, axis=0)
        modes = np.stack(modes, axis=0)
        masks = np.stack(masks, axis=0)
        return (images_SPIOBSID, mads, modes, masks)

    def error_histo_plotting(self, data: tuple[np.ndarray], paths: dict[str, str], error_masks: np.ndarray, SPIOBSID_files: list[str], filenames_interval: tuple[list[str], list[str]],
                             SPIOBSID: str, detector: int) -> None:

        # Initialisation
        before_filenames, after_filenames = filenames_interval
        all_filenames = before_filenames + after_filenames

        images_SPIOBSID, mads, modes, _ = data

        images_before = np.stack([
            fits.getdata(SpiceUtils.ias_fullpath(filename), detector)[0, :, :, 0].astype('float64')
            for filename in before_filenames
        ], axis=0) if before_filenames != [] else None
        images_after = np.stack([
            fits.getdata(SpiceUtils.ias_fullpath(filename), detector)[0, :, :, 0].astype('float64')
            for filename in after_filenames
        ], axis=0) if after_filenames != [] else None

        width, rows, cols = np.where(error_masks)

        # Randomly choosing a section of the errors 
        num_to_keep = int(self.plot_ratio * len(width)) + 1 
        indices = np.random.choice(len(width), num_to_keep if num_to_keep < len(width) else len(width) , replace=False)
        reduced_w = width[indices]
        reduced_r = rows[indices]
        reduced_c = cols[indices]

        processed_vals = set()
        for w, r, c in zip(reduced_w, reduced_r,  reduced_c):
            if (r, c) in processed_vals: continue
            processed_vals.add((r, c))
            filename = SPIOBSID_files[w]
            name_dict = SpiceUtils.parse_filename(filename)
            date = parse_date(name_dict['time'])

            # Getting the corresponding pixel data
            pixel_before = images_before[:, r, c] if images_before is not None else None
            pixel_after = images_after[:, r, c] if images_after is not None else None
            pixel_now = images_SPIOBSID[w, r, c]
            # Getting the stats
            mode_pixel = modes[w, r, c]
            mad_pixel = mads[w, r, c]

            # ERROR HISTOGRAM
            hist_name = f'Errors_ID{SPIOBSID}_w{w}_r{r}_c{c}.png'
            bins = self.binning(images_SPIOBSID[:, r, c])
            plt.hist(images_SPIOBSID[:, r, c], color='green', bins=bins, label="Same ID data", alpha=0.5)
            plt.title(f'Histogram, tot {len(all_filenames) + 1}, same ID {len(SPIOBSID_files)}, date {date.year:04d}-{date.month:02d}', fontsize=12)
            if pixel_before is not None:
                bins = self.binning(pixel_before)
                plt.hist(pixel_before, bins=bins, histtype='step', edgecolor=(0.8, 0.3, 0.3, 0.6))
                plt.hist(pixel_before, bins=bins, label='Main data before acquisition', color=(0.8, 0.3, 0.3, 0.2))     
            if pixel_after is not None:
                bins = self.binning(pixel_after)
                plt.hist(pixel_after, bins=bins, histtype='step', edgecolor=(0, 0.3, 0.7, 0.6))
                plt.hist(pixel_after, bins=bins, label='Main data after acquisition', color=(0, 0.3, 0.7, 0.2))
            plt.axvline(mode_pixel, color='magenta', linestyle='--', label='Used mode')
            plt.axvline(mode_pixel + self.coef * mad_pixel, color='magenta', linestyle=':', label='Clipping value')
            bins = self.binning(images_SPIOBSID[:, r, c])
            plt.hist(pixel_now, bins=bins, label='Studied acquisition', histtype='step', edgecolor='black')
            plt.xlabel('Detector count', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend()
            plt.savefig(os.path.join(paths['histograms'], hist_name), bbox_inches='tight', dpi=300)
            plt.close()

    def time_integration(self, time_interval: int, filenames: list[str], SPIOBSID_files: list[str]) -> tuple[list[str], list[str]] | None:

        # Initialisation
        name_dict = SpiceUtils.parse_filename(SPIOBSID_files[0])
        date = parse_date(name_dict['time'])  # deciding that the files with the same SPIOBSID were taken at pretty much the same time

        year_max = date.year
        year_min = date.year
        month_max = date.month + int(time_interval / 2)
        month_min = date.month - int(time_interval / 2)

        if month_max > 12:
            year_max += (month_max - 1) // 12
            month_max = month_max % 12
        if month_min < 1:
            year_min -= (abs(month_min) // 12) + 1
            month_min = 12 - (abs(month_min) % 12)

        date_max = f'{year_max:04d}{month_max:02d}{date.day:02d}T{date.hour:02d}{date.minute:02d}{date.second:02d}'
        date_min = f'{year_min:04d}{month_min:02d}{date.day:02d}T{date.hour:02d}{date.minute:02d}{date.second:02d}'
        date = f'{date.year:04d}{date.month:02d}{date.day:02d}T{date.hour:02d}{date.minute:02d}{date.second:02d}' # TODO: need to make sure that the date values haven't changed

        filenames_interval = []
        for filename in filenames:
            name_time = SpiceUtils.parse_filename(filename)['time']
            if name_time < date_min:
                continue
            elif name_time > date_max:
                continue
            elif filename in SPIOBSID_files:
                continue
            filenames_interval.append(filename)  # only has filenames in the interval but without the same SPIOBSID, so even the file being treated isn't here 
        
        before_filenames = []
        after_filenames = []
        for filename in filenames_interval:
            filename_time = SpiceUtils.parse_filename(filename)['time']

            if filename_time < date:
                before_filenames.append(filename)
            else:
                after_filenames.append(filename)
        before_filenames = before_filenames if not isinstance(before_filenames, str) else [before_filenames]
        after_filenames = after_filenames if not isinstance(after_filenames, str) else [after_filenames]
        used_filenames = (before_filenames, after_filenames)  # so it is a tuple[list[str], list[str]] 
        # used_filenames therefore represents the filenames lists used before and after the treated darks (without using the other same SPIOBSID files)
        return used_filenames
    
    def unique_datadict(self, time_interval: int, exposure: float, detector: int, SPIOBSID_files: list[str], detections: np.ndarray, errors: np.ndarray, ratio: np.ndarray, weights_tot: np.ndarray, weights_error: np.ndarray, 
                        weights_ratio: np.ndarray, filenames_interval: tuple[list[str], list[str]]) -> pd.DataFrame:
        """
        Function to create a dictionary containing some useful information on each exposure times. This is
        done to save the stats in a csv file when the code finishes running.
        """

        before_filenames, after_filenames = filenames_interval
        all_filenames = before_filenames + after_filenames

        # Initialisation
        name_dict = SpiceUtils.parse_filename(SPIOBSID_files[0])
        date = parse_date(name_dict['time'])

        # Creation of the stats
        group_nb = len(SPIOBSID_files)
        group_date = f'{date.year:04d}{date.month:02d}{date.day:02d}'
        SPIOBSID = name_dict['SPIOBSID']
        tot_detection = np.sum(detections)
        tot_error = np.sum(errors)
        tot_ratio = tot_error / tot_detection if tot_detection != 0 else np.nan

        # Corresponding lists or arrays
        times, a, b, c, d, e, f, g, h, i = np.full((group_nb, 10), [time_interval, exposure, detector, group_date, group_nb,
                                                                tot_detection, tot_error, tot_ratio, SPIOBSID, len(all_filenames) + 1]).T

        data_dict = {
            'Time interval [months]': times, 'Exposure time [s]': a, 
            'Detector': b, 'Group date': c,
            'Nb of files with same ID': d, 'Tot nb of detections': e, 
            'Tot nb of errors': f, 'Ratio errors/detections': g, 
            'Filename': SPIOBSID_files, 'SPIOBSID': h, 
            'Nb of used images': i, 'Nb of detections': detections, 
            'Nb of errors': errors, 'Ratio': ratio,
            'Weighted detections': weights_tot, 'Weighted errors': weights_error,
            'Weighted ratio': weights_ratio,
        }
        return pd.DataFrame(data_dict)

    def stats(self, data: tuple[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        To calculate some stats to have an idea of the efficacy of the method. The output is a set of masks
        giving the positions where the method outputted a worst result than the initial image.
        """

        # Initialisation
        images_SPIOBSID, _, modes, masks = data

        # For the stats
        new_images_SPIOBSID = np.copy(images_SPIOBSID)
        data_med = np.median(images_SPIOBSID, axis=0).astype('float64')
        meds_dif = images_SPIOBSID - data_med

        # Difference between the end result and the initial one
        new_images_SPIOBSID[masks] = modes[masks]
        new_meds_dif = new_images_SPIOBSID - data_med

        # Creating a new set of masks that shows where the method made an error
        new_masks = np.abs(new_meds_dif) > np.abs(meds_dif)

        ### MAIN STATS
        # Initialisation of the corresponding matrices
        weights_errors = np.zeros(np.shape(images_SPIOBSID))
        weights_errors[new_masks] = np.abs(np.abs(meds_dif[new_masks]) - np.abs(new_meds_dif[new_masks]))
        weights_tots = np.abs(np.abs(meds_dif) - np.abs(new_meds_dif))
        # Calculating the number of detections and errors per dark
        detections = np.sum(masks, axis=(1, 2))
        errors = np.sum(new_masks, axis=(1, 2))
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
            ratio = np.stack(ratio, axis=0)
            weights_ratio = np.stack(weights_ratio, axis=0)
        return new_masks, detections, errors, ratio, weights_tot, weights_error, weights_ratio

    def binning(self, data: np.ndarray) -> np.ndarray:
        """
        Small function to calculate the appropriate bin count.
        """

        try:
            left_border = (np.min(data) // self.bins * self.bins) 
            right_border = (np.max(data) // self.bins * self.bins) 
            return np.arange(left_border, right_border + 0.1, self.bins)
        except Exception as e:
            if self.verbose > 0: print(f'Bins set to 1 as Exception: {e}')
            return 1
    
    def average_statistics(self):
        paths = self.paths()

        mainfile_path = os.path.join(paths['main'], 'All_errors.csv')
        pandas_alldata = pd.read_csv(mainfile_path)

        avgs_used_images = []
        time_intervals = []
        tot_detections = []
        tot_errors = []
        tot_ratios = []
        tot_detections_weighted = []
        tot_errors_weighted = []
        tot_ratios_weighted = []
        pandas_intervals = pandas_alldata.groupby('Time interval [months]')

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
            pandas_exposures = dataframe_time.groupby('Exposure time [s]')
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

            exp_name = f'Errors_summary_inter{key}.csv'
            exp_path = os.path.join(paths['main'], f'{key}months')
            exp_dict = {'Time interval [months]': np.full(len(exps), key), 'Exposure time [s]': exps,
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

        inter_dict = {'Time interval [months]': time_intervals, 'Avg used images': avgs_used_images, 'Nb detections': tot_detections,
                    'Nb errors': tot_errors, 'Ratio': tot_ratios, 'Weighted detections': tot_detections_weighted,
                    'Weighted errors': tot_errors_weighted, 'Weighted ratio': tot_ratios_weighted}
        nw_pandas_inter = pd.DataFrame(inter_dict)
        inter_name = f'All_errors_main_summary.csv'
        nw_pandas_inter.to_csv(os.path.join(paths['main'], inter_name), index=False)


if __name__ == '__main__':

    import sys
    print(f'Python version is {sys.version}')
    # test = CosmicRemoval(verbose=1, time_intervals=[6], statistics=True, processes=64, flush=True)
    test = CosmicRemovalStatsPlots(verbose=1, processes=64, time_intervals=[6], statistics=True, plots=True, plot_ratio=0.01, flush=True)

