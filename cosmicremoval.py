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
from dateutil.parser import parse as parse_date
from multiprocessing.managers import BaseManager as SyncManager
from multiprocessing.queues import Queue as QUEUE
from multiprocessing import Process, Manager

# Local Python files
from Common import RE, SpiceUtils, Decorators, MultiProcessing, Pandas

# Filtering initialisation
cat = SpiceUtils.read_spice_uio_catalog()
filters = cat.STUDYDES.str.contains('dark') & (cat['LEVEL'] == 'L1')
init_res = cat[filters]

# TODO: need to check if for the actual processing I am not using the actual acquisition for the mad and mode calculations

class ParentFunctions:
    """Parent class to the Cosmic removal classes as I have created two really similar classes. The first for actually treating the single darks and the
    second to plot and compute some statistics on the multi-darks (to have an idea of the efficiency of the cosmic removal).
    Therefore, this class is only here to store the common functions to both the aforementioned classes.
    """

    def __init__(self, fits: bool, statistics: bool, plots: bool, verbose: int):
        """The initialisation of the class.

        Args:
            fits (bool): if True, then cosmic treated FITS files are created. IMPORTANT: The files will have a wrong filename and header info...
            statistics (bool): if True, then statistics are saved in .csv files. 
            plots (bool): if True, then histograms of the errors are saved as .png files. Only works for the non single-darks.
            verbose (int): dictates the precision of the outputted prints related to code running time. The higher the number (3 being the max), the higher
            the number of outputted prints.
        """
        
        # Shared attributes
        self.fits = fits
        self.making_statistics = statistics
        self.making_plots = plots
        self.verbose = verbose

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

        paths = {}
        initial_paths = {'main': os.path.join(os.getcwd(), 'statisticsAndPlots_processed_fits_no_minfilenb')}
        if self.fits: initial_paths['fits'] = os.path.join(initial_paths['main'], 'fits')

        if time_interval != -1:
            initial_paths['time_interval'] = os.path.join(initial_paths['main'], f'Date_interval{time_interval}')

            if exposure != -0.1:
                initial_paths['exposure'] = os.path.join(initial_paths['time_interval'], f'Exposure{exposure}')

                if detector != -1:
                    initial_paths['detector'] = os.path.join(initial_paths['exposure'], f'Detector{detector}')

                    # Secondary paths
                    directories = []
                    if self.making_statistics: directories.append('Statistics')
                    if self.making_plots: directories.append('Histograms')

                    paths = {directory: os.path.join(initial_paths['detector'], directory) for directory in directories}
        all_paths = {}
        for d in [initial_paths, paths]:
            for _, path in d.items(): os.makedirs(path, exist_ok=True)
            all_paths.update(d)
        return all_paths

    def exposure(self):
        """Function to find the different exposure times in the SPICE catalogue.

        Returns:
            list[float]: exposure times that will be processed.
        """

        # Getting the exposure times and nb of occurrences
        exposure_counts = Counter(init_res.XPOSURE)
        exposure_weighted = np.stack(list(exposure_counts.items()), axis=0)
        # occurrences_filter = (exposure_weighted[:, 1] > self.min_filenb)
        # self.exposures = exposure_weighted[occurrences_filter][:, 0]
        self.exposures = exposure_weighted[:, 0]

        # Printing the values
        if self.verbose > 0: 
            for exposure in exposure_weighted: print(f'For exposure time {exposure[0]}s there are {int(exposure[1])} darks.') 
            # print(f'\033[93mExposure times with less than \033[1m{self.min_filenb}\033[0m\033[93m darks are not kept.\033[0m')
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
            total_darks = sorted_dict['Total number of darks'].sum().round()
            total_row = pd.DataFrame({
                'Exposure time (s)': ['Total'], 
                'Total number of darks': [total_darks],
            })
            sorted_dict = pd.concat([sorted_dict, total_row], ignore_index=True)
            sorted_dict['Total number of darks'] = sorted_dict['Total number of darks'].astype('int')
            sorted_dict.to_csv(os.path.join(paths['main'], csv_name), index=False)
    
    def images_all(self, exposure: float) -> list[str] | list[None]:
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
        for files in filenames:
            # Opening the files
            header = fits.getheader(SpiceUtils.ias_fullpath(files), 0)

            if header['BLACKLEV'] == 1: left_nb += 1; continue
            if 'glow' in header['OBS_DESC'].lower(): weird_nb += 1; continue # Weird "glow" darks
            if header['T_SW'] > 0 and header['T_LW'] > 0: a += 1; continue

            # if 'PRSTEP6' in header: print(f"\033[31mfile already treated with {header['PRSTEP6']}\033[0m", flush=True); continue
            if header['V_GAPSW'] > 2000: high_voltage += 1; continue
            # if 'Extended' in header['OBS_DESC']: print(f"\033[31mEXTENDED IN HEADER, FILE {files}\033[0m")
            # if 'background' in header['OBS_DESC']: print(f"\033[31mBACKGROUND IN HEADER, FILE {files}\033[0m")

            all_files.append(files)

        if self.verbose > 1:
            if a != 0: print(f'\033[31mExp{exposure} -- Tot nb files with high temp: {a}\033[0m')
            if left_nb != 0: print(f'\033[31mExp{exposure} -- Tot nb files with bias subtraction: {left_nb}\033[0m')
            if weird_nb != 0: print(f'\033[31mExp{exposure} -- Tot nb of "weird" files: {weird_nb}\033[0m')
            if high_voltage != 0: print(f'\033[31mExp{exposure} -- Tot nb of "high voltage" files: {high_voltage}\033[0m')

        if self.verbose > 0: print(f'Exp{exposure} -- Nb of "usable" files: {len(all_files)}')

        if len(all_files) == 0: 
            if self.verbose > 0: print(f'\033[91m ERROR: NO "USABLE" ACQUISITIONS FOR EXP{exposure}- CHANGING EXPOSURE\033[0m')
            return [None]
        return all_files
    
    def mode_along_axis(self, arr: np.ndarray) -> int:
        """Outputs the mode of a given binned array.

        Args:
            arr (np.ndarray): the pre-binned data array.

        Returns:
            int: the mode of the array.
        """

        return np.bincount(arr.astype('int32')).argmax()

    def samedarks(self, filenames: list[str]) -> dict[str, str]:
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
    def __init__(self, processes: int = 128, max_date: str | None = '20230402T030000', coefficient: int | float = 6, min_filenb: int = 20, 
                 set_min: int = 4, time_interval: int = 6, statistics: bool = False, bins: int = 5, verbose: int = 1, flush: bool = False):
        """To initialise but also run the CosmicRemoval class.

        Args:
            processes (int, optional): sets the maximum number of processes for the multi-processing. Defaults to 10.
            max_date (str | None, optional): the maximum date for which the treatment is done on all darks. Defaults to '20230402T030000'.
            coefficient (int | float, optional): the m.a.d. multiplication coefficient that decides what is flagged as a cosmic collision. Defaults to 6.
            min_filenb (int, optional): the minimum number of files needed in a specific exposure time set to start the treatment. Defaults to 20.
            set_min (int, optional): the minimum number of files need to consider a SPIOBSID set as single darks. Defaults to 6.
            time_interval (int, optional): the time interval considered in months (e.g. 6 is 3 months prior and after each acquisition). Defaults to 6.
            statistics (bool, optional): if True, then statistics are saved in .csv files. Defaults to False.
            bins (int, optional): the binning value in detector counts for the histograms. Defaults to 5.
            verbose (int, optional): decides how precise you want your logging prints to be, 0 being not prints and 2 being the maximum. Defaults to 1.
            flush (bool, optional): sets the flush argument of some of the main print statements. Defaults to False.
        """
    
        super().__init__(True, False, False, verbose)
        # Arguments
        self.processes = processes
        self.max_date = max_date if max_date is not None else '30000100T000000' 
        self.coef = coefficient
        self.min_filenb = min_filenb
        self.set_min = set_min
        self.time_interval = time_interval 
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
        self.multiprocess()

    @Decorators.running_time
    def multiprocess(self) -> None:
        """To run multiple processes if multiprocessing=True.
        """

        # Choosing to multiprocess or not
        if self.multiprocessing:
            # Setting up the multiprocessing 
            manager = Manager()
            queue = manager.Queue()
            indexes = MultiProcessing.Pool_indexes(len(self.exposures), self.processes)

            if self.verbose > 0: print(f'Number of processes used is: {len(indexes) * (self.processes // len(self.exposures))}')

            processes = [Process(target=self.main, args=(index, queue)) for index in indexes]            
            for p in processes: p.start()
            for p in processes: p.join()

            results = []
            while not queue.empty(): results.append(queue.get())
        else:
            results = self.main((0, len(self.exposures) - 1))

        if self.making_statistics: self.saving_main_numbers(results)
    
    @Decorators.running_time
    def main(self, index: tuple[int, int], queue: QUEUE | None = None) -> list[tuple[float, list[str]]] | None:
        """Main structure of the code after the multiprocessing. Does the treatment given an exposure time and outputs the results.

        Args:
            index (tuple[int, int]): the data's start and end index for each process. 
            queue (QUEUE | None): a multiprocessing.Manager.Queue() object. Defaults to None.

        Returns:
            None | tuple[float, list[str]]: contains the exposure time and the corresponding filenames list if not multiprocessing. Else,
            just populates the queue with those values.
        """        

        if self.verbose > 2: print(f'The process id is {os.getpid()}')
        if not self.multiprocessing: results = [None] * len(self.exposures)

        # Deciding to also set multiprocessing if needed
        processes_needed = self.processes // len(self.exposures)
        multiprocessing = self.multiprocessing and (processes_needed > 1)
        paths = self.paths()

        for i, exposure in enumerate(self.exposures[index[0]:index[1] + 1]):
            filenames = self.images_all(exposure)
            
            # if len(filenames) < self.min_filenb:
            #     if self.verbose > 0: print(f'\033[91mInter{self.time_interval}_exp{exposure} -- Less than {self.min_filenb} usable files. Changing exposure times.\033[0m')
            #     return None

            # MULTIPLE DARKS analysis
            same_darks = self.samedarks(filenames)
            kwargs = {
                'exposure': exposure,
                'filenames': filenames,
                'same_darks': same_darks,
                'paths': paths,
            }

            if multiprocessing:
                # Multiprocessing
                manager = Manager()
                sub_queue = manager.Queue()
                indexes = MultiProcessing.Pool_indexes(len(filenames), processes_needed)

                processes = [
                    Process(target=self.exposure_loop, kwargs={'index': index, 'queue': sub_queue, 'position': pos, **kwargs}) 
                    for pos, index in enumerate(indexes)
                ]
                for p in processes: p.start()
                for p in processes: p.join()

                sub_results = [None] * processes_needed
                while not sub_queue.empty():
                    identifier, result = sub_queue.get()
                    sub_results[identifier] = result
                processed_filenames = [filename for filename_list in sub_results if filename_list is not None for filename in filename_list]
            else:
                processed_filenames = self.exposure_loop(index=(0, len(filenames) - 1), **kwargs)    
            
            if self.verbose > 0: print(f'\033[1;33mFor exposure {exposure}, {len(processed_filenames)} files processed\033[0m')

            if self.multiprocessing: 
                queue.put((exposure, processed_filenames))
            else:
                results[i] = (exposure, processed_filenames)
        if not self.multiprocessing: return results

    def exposure_loop(self, exposure: float, filenames: list[str], same_darks: dict[str, str], paths: dict[str, str], index: tuple[int, int], 
                      queue: QUEUE | None = None, position: int | None = None) -> list[str] | None:
        """To loop over the filenames for each exposure times. Gives out the corresponding treated filenames with the exposure time used.

        Args:
            exposure (float): the exposure time used.
            filenames (list[str]): the list of all the filenames in that exposure time.
            same_darks (dict[str, str]): dictionary with .items() = (SPIOBSID, corresponding_dark_filenames).
            paths (dict[str, str]): the path to the needed directories.
            index (tuple[int, int]): the start and end index for each processes.
            queue (QUEUE | None, optional): a multiprocessing.Manager.Queue() object. Defaults to None.
            position (int | None, optional): the index representing the position in the parent loop. Defaults to None.

        Raises:
            ValueError: if the filenames doesn't match the usual SPICE FITS filenames.
            ValueError: if there are no NANs in the acquisitions but the header information said there are.
            ValueError: if there are NANs in the acquisitions but the header information didn't say so.

        Returns:
            list[str] | None: list of the processed filenames if not multiprocessing this function. Else, just populates the queue object.
        """
        
        processed_filenames = []
        for loop, filename in enumerate(filenames[index[0]:index[1] + 1]):
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
                    if self.verbose > 1: print(f'\033[31mExp{exposure}_imgnb{loop} -- {length} images in SPIOBSID. Going to the next acquisition.\033[0m')
                    continue
                elif self.verbose > 1:
                    print(f'Image from a SPIOBSID set of {length} darks but before May 2023.')
            
            interval_filenames = self.time_integration(filename, filenames)
            if len(interval_filenames) < self.set_min:
                if self.verbose > 1: print(f'\033[31mExp{exposure}_imgnb{loop} -- Set is only of {self.set_min} files. Going to the next filename.\033[0m')
                continue

            processed_filenames.append(filename)                

            check = False
            new_images = []
            for detector in range(2):
                if os.path.exists(os.path.join(SpiceUtils.ias_fullpath(filename))):
                    image = fits.getdata(SpiceUtils.ias_fullpath(filename), detector)[0, :, :, 0].astype('float64')  #TODO: need to check if I need float64 and if np.stack has a dtype argument
                else:
                    if self.verbose > 1: print("filename doesn't exist, adding a +1 to the version number", flush=self.flush)
                    image = fits.getdata(SpiceUtils.ias_fullpath(new_filename), detector)[0, :, :, 0].astype('float64')
                mad, mode = self.mad_mode(interval_filenames, detector)
                mask = image > self.coef * mad + mode     

                nw_image = np.copy(image)  # TODO: need to check why np.copy is not the preferred copy method.
                nw_image[mask] = mode[mask]
                nw_image = nw_image[np.newaxis, :, :, np.newaxis]
                if np.isnan(image).any():
                    if self.verbose > 1: print(f'\033[1;94mImage contains nans: {np.isnan(nw_image).any()}\033[0m')
                    nw_image[np.isnan(nw_image)] = 65535 
                    check = True
                new_images.append(nw_image)
            new_images = np.stack(new_images, axis=0).astype('uint16')

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
            hdul_new.writeto(os.path.join(paths['fits'], new_filename), overwrite=True)

            if self.verbose > 2: print(f'File {filename} processed.', flush=self.flush)

        if queue is None: return processed_filenames
        queue.put((position, processed_filenames if processed_filenames != [] else None))

    def saving_main_numbers(self, results: list[tuple[float, list[str]]]) -> None:
        """Saving the number of files processed and which SPIOBSID were processed.

        Args:
            results (list[tuple[float, list[str]]]): the results gotten from running the code as tuples 
            showing the exposure time value and the corresponding list of processed filenames.
        """

        paths = self.paths()

        # Filenames csv
        filenames_df = pd.DataFrame([filename for _, filenames in results for filename in filenames], columns=['Filenames'])
        filenames_df.sort_values(by='Filenames')
        filenames_name = 'Processed_filenames.csv'
        filenames_df.to_csv(os.path.join(paths['main'], filenames_name), index=False)

        # Processed nb of files csv
        processed_nb_df = pd.DataFrame([(exposure, len(filenames)) for exposure, filenames in results], columns=['Exposure time (s)', 'Nb of processed files'])
        total_processed = processed_nb_df['Nb of processed files'].sum()
        total_processed_df = pd.DataFrame({
            'Exposure time (s)': ['Total'],
            'Nb of processed files': [total_processed],
        })
        processed_nb_df = pd.concat([processed_nb_df, total_processed_df], ignore_index=True)
        df_name = 'Total_darks_info.csv'
        df = pd.read_csv(os.path.join(paths['main'], df_name))
        # Making sure the same column values properly match
        safe_round_kwargs = {
            'decimals': 1,
            'try_convert_string': True,
            'verbose': self.verbose,
        }
        df['Exposure time (s)'] = df['Exposure time (s)'].apply(Pandas.safe_round, **safe_round_kwargs)
        processed_nb_df['Exposure time (s)'] = processed_nb_df['Exposure time (s)'].apply(Pandas.safe_round, **safe_round_kwargs)
        result_df = df.merge(processed_nb_df, on='Exposure time (s)', how='outer')
        result_df['Nb of processed files'] = result_df['Nb of processed files'].astype('Int16')
        result_df.to_csv(os.path.join(paths['main'], df_name), index=False)
        
    def time_integration(self, filename: str, files: list[str | None]) -> list[str]:
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


class CosmicRemovalStatsPlots(ParentFunctions):
    """To get a visual and statistical idea of the performance of the method. Here, only darks that were taken as a group (i.e. multiple darks with same SPIOBSID) are taken into account to try and flag any
    false detections. The data used to treat each file doesn't include the files that also have the same SPIOBSID (to try and minimize bias). 

    Args:
        ParentFunctions (_type_): To store functions that can be used in this and the CosmicRemoval class.

    Returns: saves some error histogram images and csv files related to the error prediction.
    """

    @typechecked
    def __init__(self, processes: int = 121, coefficient: int | float = 6, min_filenb: int = 20, set_min: int = 4, time_intervals: list[int] = [6], bins: int = 5, statistics: bool = True, plots: bool = True, 
                plot_ratio: float = 0.01, verbose: int = 0, flush: bool = False, circumcision: bool = False):
        """Initialisation of the CosmicRemovalStatsPlots class.

        Args:
            processes (int, optional): _description_. Defaults to 128.
            coefficient (int | float, optional): _description_. Defaults to 6.
            min_filenb (int, optional): _description_. Defaults to 20.
            set_min (int, optional): _description_. Defaults to 4.
            time_intervals (list[int], optional): _description_. Defaults to [6].
            bins (int, optional): _description_. Defaults to 5.
            statistics (bool, optional): _description_. Defaults to True.
            plots (bool, optional): _description_. Defaults to True.
            verbose (int, optional): _description_. Defaults to 0.
            flush (bool, optional): _description_. Defaults to False.
        """
        
        super().__init__(False, statistics, plots, verbose)
        # Arguments
        self.processes = processes
        self.coef = coefficient
        self.min_filenb = min_filenb
        self.set_min = set_min
        self.time_intervals = time_intervals
        self.bins = bins
        self.plot_ratio = plot_ratio
        self.flush = flush
        self.circumcision = circumcision

        # New class attributes
        self.multiprocessing = True if processes > 1 else False

        # Code functions
        self.exposure()
        self.main()

    @Decorators.running_time
    def main(self) -> None:
        """Function for multiprocessing if self.processes > 1. No multiprocessing done otherwise.
        """

        args = list(product(self.time_intervals, self.exposures))
        data_pandas_interval = []

        # Choosing to multiprocess or not
        if self.multiprocessing:
            # Multiprocessing set-up
            manager = Manager()
            queue_input = manager.Queue()
            queue_output = manager.Queue()
            nb_processes = min(self.processes, len(args))
            sub_nb_processes = self.processes // len(args)
            sub_nb_processes = sub_nb_processes if sub_nb_processes > 1 else 1

            if self.verbose > 0: print(f'Max nb of used processes is {nb_processes * sub_nb_processes}')

            # Setting up the input queue
            for argument in args: queue_input.put(argument)
            for _ in range(nb_processes): queue_input.put(None)
            # Starting the multiprocessing
            processes = [Process(target=self.each_exposure, args=(sub_nb_processes, queue_input, queue_output)) for _ in range(nb_processes)]
            for p in processes: p.start()
            for p in processes: p.join()
            # Getting the results
            while not queue_output.empty(): data_pandas_interval.append(queue_output.get())
        else:
            for argument in args: 
                data_frame = self.each_exposure(processes_needed=1, args=argument)
                if self.making_statistics:data_pandas_interval.append(data_frame)

        if self.making_statistics: 
            self.saving_csv(data_pandas_interval)
            self.average_statistics()

    @Decorators.running_time
    def each_exposure(self, processes_needed: int, queue_input: QUEUE | None = None, queue_output: QUEUE | None = None, args: tuple[int, float] | None = None) -> pd.DataFrame | None:
        """_summary_

        Args:
            processes_needed (int): _description_
            queue_input (QUEUE | None, optional): _description_. Defaults to None.
            queue_output (QUEUE | None, optional): _description_. Defaults to None.
            args (tuple[int, float] | None, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame | None: _description_
        """

        while True:
            if self.multiprocessing:
                # Fetching the inputs
                item = queue_input.get()
                if item is None: break
                time_interval, exposure = item
            else:
                time_interval, exposure = args

            filenames = self.images_all(exposure)

            # if len(filenames) < self.min_filenb: 
            #     if self.verbose > 0: print(f'\033[91mInter{time_interval}_exp{exposure} -- Less than {self.min_filenb} usable files. Changing exposure times.\033[0m', flush=self.flush)
            #     return None

            # MULTIPLE DARKS analysis
            same_darks = self.samedarks(filenames)

            # Deciding to also set multiprocessing if needed
            multiprocessing = self.multiprocessing and (processes_needed > 1)

            if multiprocessing:
                # Setup
                manager = Manager()
                sub_queue_input = manager.Queue()
                sub_queue_output = manager.Queue()
                nb_processes = min(len(same_darks), processes_needed)

                #Setting up the input queue
                for SPIOBSID in same_darks: sub_queue_input.put(SPIOBSID)
                for _ in range(nb_processes): sub_queue_input.put(None)
                # Starting the multiprocessing
                kwargs = {
                    'queue_input': sub_queue_input,
                    'queue_output': sub_queue_output,
                    'same_darks': same_darks,
                    'filenames': filenames,
                    'time_interval': time_interval,
                    'exposure': exposure,
                    'multiprocessing': True,
                }
                processes = [Process(target=self.each_exposure_sub, kwargs=kwargs) for _ in range(nb_processes)]
                for p in processes: p.start()
                for p in processes: p.join()
                # Getting the results
                results = []
                while not sub_queue_output.empty(): results.append(sub_queue_output.get())
                data_pandas_exposure = pd.concat(results, ignore_index=True)
            else:
                kwargs = {
                    'same_darks': same_darks,
                    'filenames': filenames,
                    'time_interval': time_interval,
                    'exposure': exposure,
                    'multiprocessing': False,
                }
                results = [None] * len(same_darks) 
                for i, SPIOBSID in enumerate(same_darks): results[i] = self.each_exposure_sub(SPIOBSID=SPIOBSID, **kwargs)
                data_pandas_exposure = pd.concat(results, ignore_index=True)
            
            if not self.making_statistics:
                return
            elif not self.multiprocessing:
                return data_pandas_exposure
            queue_output.put(data_pandas_exposure)
    
        if self.verbose > 1: print(f'Inter{time_interval}_exp{exposure} -- Chunks finished and histogram plotting done.', flush=self.flush)

    def each_exposure_sub(self, same_darks: dict[str, str], filenames: list[str], time_interval: int, exposure: float, queue_input: QUEUE | None = None, 
                          queue_output: QUEUE | None = None, multiprocessing: bool = False, SPIOBSID: str | None = None) -> pd.DataFrame | None:
        while True:
            if multiprocessing:
                # Fetching the inputs
                item = queue_input.get()
                if item is None: return
                SPIOBSID = item

            for detector in range(2):
                if self.making_statistics: data_pandas_detector = pd.DataFrame()

                # Path initialisation
                paths = self.paths(time_interval=time_interval, exposure=exposure, detector=detector)

                files = same_darks[SPIOBSID]
                if len(files) < 3: continue
                first_file = files[0]
                date_dict = SpiceUtils.parse_filename(first_file)
                date = parse_date(date_dict['time'])
                date = f'{date.year:04d}{date.month:02d}{date.day:02d}T{date.hour:02d}{date.minute:02d}{date.second:02d}'

                if self.circumcision and (date > '20240303T000000'): continue

                # Getting the filenames for each time integration
                kwargs = {
                    'time_interval': time_interval,
                    'exposure': exposure,
                    'detector': detector,
                    'filenames': filenames,
                    'SPIOBSID': SPIOBSID,
                    'same_darks': same_darks,
                }
                used_filenames = self.time_integration(**kwargs)
                if used_filenames is None: continue

                # Getting the treatment data for each same SPIOBSID file
                kwargs = {
                    'detector': detector,
                    'files': files,
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
                    'same_darks': same_darks,
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
                    'files': files,
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
                data_pandas.to_csv(os.path.join(paths['Statistics'], csv_name), index=False)
                data_pandas_detector = pd.concat([data_pandas_detector, data_pandas], ignore_index=True)
   
            if self.making_statistics:
                if not multiprocessing: return data_pandas_detector
                queue_output.put(data_pandas_detector) 
            elif not multiprocessing:
                return                 

    def saving_csv(self, data_list: list[pd.DataFrame]) -> None:
        last_time = 0
        first_try = 0
        for loop, pandas_dict in enumerate(data_list):
            if pandas_dict.empty: continue
            time_interval = pandas_dict['Time interval [months]'][0]
            exposure = pandas_dict['Exposure time [s]'][0]
            paths = self.paths(time_interval=time_interval, exposure=exposure)

            # Saving a csv file for each exposure time
            csv_name = f'Alldata_inter{time_interval}_exp{exposure}.csv'
            pandas_dict.to_csv(os.path.join(paths['exposure'], csv_name), index=False)
            if self.verbose > 1: print(f'Inter{time_interval}_exp{exposure} -- CSV files created')

            if time_interval == last_time:
                pandas_inter = pd.concat([pandas_inter, pandas_dict], ignore_index=True)
                if loop == len(data_list) - 1:
                    pandas_name0 = f'Alldata_inter{time_interval}.csv'
                    pandas_inter.to_csv(os.path.join(paths['time_interval'], pandas_name0), index=False)
                    if self.verbose > 0: print(f'Inter{time_interval} -- CSV files created')
            else:
                if first_try != 0:
                    paths = self.paths(time_interval=last_time)
                    pandas_name0 = f'Alldata_inter{last_time}.csv'
                    pandas_inter.to_csv(os.path.join(paths['time_interval'], pandas_name0), index=False)
                    if self.verbose > 0: print(f'Inter{time_interval} -- CSV files created', flush=self.flush)
                first_try = 1
                last_time = time_interval
                pandas_inter = pandas_dict

        data_list = pd.concat(data_list, ignore_index=True)
        pandas_name = 'Alldata.csv'
        data_list.to_csv(os.path.join(paths['main'], pandas_name), index=False)

    def data_bundle(self, detector: int, files: list[str], filenames_interval: tuple[list[str], list[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

         # Initialisation
        before_filenames, after_filenames = filenames_interval
        all_filenames = before_filenames + after_filenames

        files_len = len(files)
        images_SPIOBSID = [None] * files_len
        mads = [None] * files_len
        modes = [None] * files_len
        masks = [None] * files_len

        for i, main_filename in enumerate(files):
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

    def error_histo_plotting(self, data: tuple[np.ndarray], paths: dict[str, str], error_masks: np.ndarray, same_darks: dict[str, str], filenames_interval: tuple[list[str], list[str]],
                             SPIOBSID: str, detector: int) -> None:

        # Initialisation
        files = same_darks[SPIOBSID] 
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
            filename = files[w]
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
            plt.title(f'Histogram, tot {len(all_filenames) + 1}, same ID {len(files)}, date {date.year:04d}-{date.month:02d}', fontsize=12)
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
            plt.savefig(os.path.join(paths['Histograms'], hist_name), bbox_inches='tight', dpi=300)
            plt.close()

    def time_integration(self, time_interval: int, exposure: float, detector: int, filenames: list[str], SPIOBSID: str, same_darks: dict[str, str]) \
        -> tuple[list[str], list[str]] | None:

        # Initialisation
        files = same_darks[SPIOBSID]
        first_filename = files[0]
        name_dict = SpiceUtils.parse_filename(first_filename)
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

        if self.circumcision:
            max_date_because_of_erros = '20240303T000000'
            date_max = date_max if date_max < max_date_because_of_erros else max_date_because_of_erros

        filenames_interval = []
        for filename in filenames:
            name_time = SpiceUtils.parse_filename(filename)['time']
            if name_time < date_min:
                continue
            elif name_time > date_max:
                continue
            elif filename in files:
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

        all_filenames = before_filenames + after_filenames

        # Checking the data length
        if len(all_filenames) < self.set_min:  
            if self.verbose > 0: print(f'\033[31mInter{time_interval}_exp{exposure}_det{detector}_ID{SPIOBSID} -- Less than {self.set_min} files. Going to next SPIOBSID\033[0m')
            return

        return used_filenames
    
    def unique_datadict(self, time_interval: int, exposure: float, detector: int, files: list[str], detections: np.ndarray, errors: np.ndarray, ratio: np.ndarray, weights_tot: np.ndarray, weights_error: np.ndarray, 
                        weights_ratio: np.ndarray, filenames_interval: tuple[list[str], list[str]]) -> pd.DataFrame:
        """
        Function to create a dictionary containing some useful information on each exposure times. This is
        done to save the stats in a csv file when the code finishes running.
        """

        before_filenames, after_filenames = filenames_interval
        all_filenames = before_filenames + after_filenames

        # Initialisation
        name_dict = SpiceUtils.parse_filename(files[0])
        date = parse_date(name_dict['time'])

        # Creation of the stats
        group_nb = len(files)
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
            'Filename': files, 'SPIOBSID': h, 
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

        mainfile_path = os.path.join(paths['main'], 'Alldata.csv')
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

            exp_name = f'Alldata_summary_inter{key}.csv'
            exp_path = os.path.join(paths['main'], f'Date_interval{key}')
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
        inter_name = f'Alldata_main_summary.csv'
        nw_pandas_inter.to_csv(os.path.join(paths['main'], inter_name), index=False)


if __name__ == '__main__':

    import sys
    print(f'Python version is {sys.version}')
    test = CosmicRemoval(verbose=1, statistics=True, processes=64)
    # test = CosmicRemovalStatsPlots(verbose=2, processes=64, statistics=True, plots=True, circumcision=False, plot_ratio=0.05)

