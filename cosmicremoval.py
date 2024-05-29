# IMPORTS
from __future__ import annotations
import os
import sys
import re

import numpy as np
import pandas as pd
import matplotlib as mpl

from astropy.io import fits
from collections import Counter
from typeguard import typechecked
from multiprocessing import Process, Manager
from dateutil.parser import parse as parse_date
from multiprocessing.queues import Queue as QUEUE

# Personnal codes
from Common import RE, SpiceUtils, Decorators


# Creating a class for the cosmicremoval
class CosmicRemoval:
    """To get the SPICE darks, separate them depending on exposure time, do a temporal analysis of the detector counts for each 
    pixels, flag any values that are too far from the distribution as cosmic rays to then replace them by the mode of the detector 
    count for said pixel.
    """

    # Finding the L1 darks
    cat = SpiceUtils.read_spice_uio_catalog()
    filters = cat.STUDYDES.str.contains('dark') & (cat['LEVEL'] == 'L1')
    res = cat[filters]

    @typechecked
    def __init__(self, multiprocessing: bool = True, max_date: str | None = '20230402T030000', coefficient: int | float = 6, min_filenb: int = 20, 
                 set_min: int = 4, time_interval: int = 6, bins: int = 5, verbose: int = 1):
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
        self.multiprocessing = multiprocessing
        self.coef = coefficient
        self.min_filenb = min_filenb
        self.set_min = set_min
        self.time_interval = time_interval  # time interval considered in months (e.g. 6 is 3 months prior and after each acquisition)
        self.bins = bins
        self.max_date = max_date if max_date is not None else '30000100T000000'  # maximum date for which the treatment is done for multi-darks
        self.verbose = verbose

        # Code functions
        self.exposures = self.Exposure()
        self.Multiprocess()

    ################################################ INITIAL functions #################################################
    def Exposure(self) -> list[float]:
        """Function to find the different exposure times in the SPICE catalogue.

        Returns:
            list[float]: exposure times that will be processed.
        """

        # Getting the exposure times and nb of occurrences
        exposure_counts = Counter(CosmicRemoval.res.XPOSURE)
        exposure_weighted = np.array(list(exposure_counts.items()))

        # Printing the values
        if self.verbose > 0: [print(f'For exposure time {exposure[0]}s there are {int(exposure[1])} darks.') for exposure in exposure_weighted]

        # Keeping the exposure times with enough darks
        occurrences_filter = (exposure_weighted[:, 1] > self.min_filenb)
        exposure_used = exposure_weighted[occurrences_filter][:, 0]

        if self.verbose > 0:
            print(f'\033[93mExposure times with less than \033[1m{self.min_filenb}\033[0m\033[93m darks are not kept.'
                f'\033[0m')
            print(f'\033[33mExposure times kept are {exposure_used}\033[0m')

        # Saving exposure stats
        csv_name = 'Nb_of_darks.csv'
        exp_dict = {'Exposure time (s)': exposure_weighted[:, 0], 'Total number of darks': exposure_weighted[:, 1]}
        pandas_dict = pd.DataFrame(exp_dict)
        sorted_dict = pandas_dict.sort_values(by='Exposure time (s)')

        total_darks = sorted_dict['Total number of darks'].sum().round()
        total_row = pd.DataFrame({
            'Exposure time (s)': ['Total'], 
            'Total number of darks': [total_darks],
        })
        sorted_dict = pd.concat([sorted_dict, total_row], ignore_index=True)
        sorted_dict.to_csv(csv_name, index=False)
        return exposure_used
    
    def Images_all(self, exposure: float) -> np.ndarray | list[None]:
        """Function to get, for a certain exposure time, the corresponding filenames.

        Args:
            exposure (float): the exposure time that is going to be studied.

        Returns:
            np.ndarray | list[None]: list of all the usable filenames found for the exposure time.
        """

        # Filtering the data by exposure time
        filters = (CosmicRemoval.res.XPOSURE == exposure)
        res = CosmicRemoval.res[filters]

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
        all_files = np.array(all_files)

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

            processes = [Process(target=self.Main, args=(queue, exposure)) for exposure in self.exposures]            
            for p in processes: p.start()
            for p in processes: p.join()

            results = []
            while not queue.empty():
                results.append(queue.get())
            self.Saving_main_numbers(results)
        else:
            data = [self.Main(exposure) for exposure in self.exposures]
            self.Saving_main_numbers(data)

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
        
    def Main(self, queue: QUEUE, exposure: float) -> None | tuple[float, int, list[str]]:
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

        print(f'The process id is {os.getpid()}')
        filename_pattern = re.compile(
                r"""
                solo
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
        
        # MAIN LOOP
        # Initialisation of the stats for csv file saving
        filenames = self.Images_all(exposure)

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
            
            pattern_match = filename_pattern.match(filename)
            if pattern_match:
                SPIOBSID = pattern_match.group('SPIOBSID')
                date = pattern_match.group('time')
                init_version = int(pattern_match.group('version'))
                new_version = init_version + 1
                new_filename = RE.replace_group(pattern_match, 'version', f'{new_version:02d}')
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

        if not self.multiprocessing: return (exposure, processed_darks_total_nb, processed_filenames)
        queue.put((exposure, processed_darks_total_nb, processed_filenames))

    def Time_interval(self, filename: str, files: np.ndarray) -> list[str]:
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

    def Samedarks(self, filenames: np.ndarray) -> dict[str, str]:
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


if __name__ == '__main__':
    mpl.rcParams['figure.figsize'] = (8, 8)

    import sys

    print(f'Python version is {sys.version}')

    test = CosmicRemoval(verbose=2)

