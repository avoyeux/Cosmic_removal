# IMPORTS
import os
import sys
import re
import common
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl

from astropy.io import fits
from collections import Counter
from typeguard import typechecked
from multiprocessing import Process, Manager
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
    def __init__(self, processes: int = 20, coefficient: int | float = 6, min_filenb: int = 20, set_min: int = 6,
                time_interval: int = 6, bins: int = 5, plots: bool = True):
        
        # Arguments
        self.processes = processes
        self.coef = coefficient
        self.min_filenb = min_filenb
        self.set_min = set_min
        self.plots = plots  # boolean to know if plots will be saved
        # self.time_intervals = time_intervals
        self.time_interval = time_interval
        self.bins = bins

        # Code functions
        self.exposures = self.Exposure()

        # Constants
        self.header_key_length = 8  # the length of the key in the headers
        self.header_value_length = 21  # the length of the value in the headers

    ################################################ INITIAL functions #################################################
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
        occurrences_filter = (exposure_weighted[:, 1] > self.min_filenb)
        exposure_used = exposure_weighted[occurrences_filter][:, 0]
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
        all_files = []
        for files in filenames:
            # Opening the files
            header = fits.getheader(common.SpiceUtils.ias_fullpath(files), 0)

            if header['BLACKLEV'] == 1:  # For on-board bias subtraction images
                left_nb += 1
                continue
            OBS_DESC = header['OBS_DESC'].lower()
            if 'glow' in OBS_DESC:  # Weird "glow" darks
                weird_nb += 1
                continue
            temp1 = header['T_SW']
            temp2 = header['T_LW']

            if temp1 > 0 and temp2 > 0:
                a += 1
                continue

            all_files.append(files)
        all_files = np.array(all_files)

        if a != 0:
            print(f'\033[31mExp{exposure} -- Tot nb files with high temp: {a}\033[0m')
        if left_nb != 0:
            print(f'\033[31mExp{exposure} -- Tot nb files with bias subtraction: {left_nb}\033[0m')
        if weird_nb != 0:
            print(f'\033[31mExp{exposure} -- Tot nb of "weird" files: {weird_nb}\033[0m')
        print(f'Exp{exposure} -- Nb of "usable" files: {len(all_files)}')
        if len(all_files) == 0:
            raise ValueError(f'\033[91m ERROR: NO "USABLE" ACQUISITIONS - STOPPING THE RUN\033[0m')
        return all_files

    ############################################### CALCULATION functions ##############################################
    @decorators.running_time
    def Multiprocess(self):
        """
        Function for multiprocessing if self.processes > 1. No multiprocessing done otherwise.
        """

        # Choosing to multiprocess or not
        if self.processes > 1:
            print(f'Number of used processes is {self.processes}', flush=True)

            # Setting up the multiprocessing 
            manager = Manager()
            queue = manager.Queue()
            processes = []
            for exposure in self.exposures:
                processes.append(Process(target=self.Main, args=(queue, exposure)))
            for p in processes:
                p.start()
            for p in processes:
                p.join()

            results = []
            while not queue.empty():
                results.append(queue.get())
            self.Saving_main_numbers(results)
        else:
            data = []
            for exposure in self.exposures:
                data.append(self.Main(exposure))
            self.Saving_main_numbers(data)

    def Saving_main_numbers(self, results):
        """
        Just saving the number of files processed and which SPIOBSID were processed.
        """

        processed_nb_df = pd.DataFrame([
            (exposure, nb)
            for exposure, nb, _ in results
            ], columns=['Exposure', 'Processed'])
        processed_nb_df.sort_values(by='Exposure')
        total_processed = processed_nb_df['Processed'].sum().round()
        total_processed_df = pd.DataFrame({
            'Exposure': ['Total'],
            'Processed': [total_processed],
        })
        processed_nb_df = pd.concat([processed_nb_df, total_processed_df], ignore_index=True)

        filenames_df = pd.DataFrame([
            filename for _, _, filename_list in results for filename in filename_list
            ], columns=['Filenames'])
        
        # Saving both stats
        df1_name = 'Nb_of_processed_darks.csv'
        processed_nb_df.to_csv(df1_name, index=False)
        df2_name = 'Processed_SPIOBSID.csv'
        filenames_df.to_csv(df2_name, index=False)
        
    @decorators.running_time
    def Main(self, queue, exposure):
        print(f'The process id is {os.getpid()}')
        # MAIN LOOP
        # Initialisation of the stats for csv file saving
        filenames = self.Images_all(exposure)

        if len(filenames) < self.min_filenb:
            print(f'\033[91mInter{self.time_interval}_exp{exposure} -- Less than {self.min_filenb} usable files. '
                  f'Changing exposure times.\033[0m')
            return

        # MULTIPLE DARKS analysis
        same_darks = self.Samedarks(filenames)

        # print(f'Inter{self.time_interval}_exp{exposure} -- Starting chunks.')
        processed_darks_total_nb = 0
        processed_SPIOBSID = []
        for loop, filename in enumerate(filenames):
            SPIOBSID = None
            for key in same_darks.keys():
                if filename in same_darks[key]:
                    SPIOBSID = key
                    break
            if len(same_darks[SPIOBSID]) > 3:
                print(f'\033[31mInter{self.time_interval}_exp{exposure}_imgnb{loop}'
                      f'-- Image from a SPIOBSID set of 4 or more darks. Going to the next acquisition.\033[0m')
                continue

            interval_filenames = self.Time_interval(filename, filenames)
            # print(f'Inter{self.time_interval}_exp{exposure}_imgnb{loop}'
            #       f' -- Nb of used files: {len(interval_filenames)}')
            
            if len(interval_filenames) < self.set_min:
                print(f'\033[31mInter{self.time_interval}_exp{exposure}_imgnb{loop} '
                      f'-- Less than {self.set_min} files for the processing. Going to the next filename.\033[0m')
                continue

            # For stats
            processed_darks_total_nb += 1
            processed_SPIOBSID.append(SPIOBSID)

            # Setting the filename for the new fits file:
            filename_pattern = re.compile(r'''(?P<group1>solo_L1_spice-n-exp_\d{8}T\d{6}_
                                V)(?P<version>\d{2})
                                (?P<group2>_\d+-\d+.fits)
                                ''', re.VERBOSE)

            matching = filename_pattern.match(filename)
            if matching:
                init_version = int(matching.group('version'))
                new_version = init_version + 1
                new_filename = f"{matching.group('group1')}{new_version:02d}{matching.group('group2')}"
            else:
                raise ValueError(f"The filename {filename} doesn't match the expected pattern.")
            
            new_images = []
            check = None
            for detector in range(2):
                mode, mad = self.Mad_mean(interval_filenames, detector)
                if os.path.exists(os.path.join(common.SpiceUtils.ias_fullpath(filename))):
                    image = np.array(fits.getdata(common.SpiceUtils.ias_fullpath(filename), detector)[0, :, :, 0], dtype='float64')
                else:
                    print("filename doesn't exist, adding a +1 to the version number")
                    image = np.array(fits.getdata(common.SpiceUtils.ias_fullpath(new_filename), detector)[0, :, :, 0], dtype='float64')
                mask = image > self.coef * mad + mode
                
                nw_image = np.copy(image)
                nw_image[mask] = mode[mask]
                nw_image = nw_image[np.newaxis, :, :, np.newaxis]
                if np.isnan(image).any():
                    print(f'\033[1;94mImage contains nans: {np.isnan(nw_image).any()}\033[0m')
                    nw_image[np.isnan(nw_image)] = 65535 
                    check = True
                new_images.append(nw_image)

            new_images = np.array(new_images, dtype='uint16')
            # print(f'Treatment for image nb {loop} done. Saving to a fits file.')

            # Headers
            init_header_SW = fits.getheader(common.SpiceUtils.ias_fullpath(filename), 0)
            init_header_LW = fits.getheader(common.SpiceUtils.ias_fullpath(filename), 1)

            if (init_header_LW['NLOSTPIX'] != 0) or (init_header_SW['NLOSTPIX'] != 0):
                if not check:
                    raise ValueError(f"There should have been a bright blue print before but didn't happen. Filename: {filename}")
            else:
                if check:
                    raise ValueError(f'Nans where found even though there are none in the headers. Filename: {filename}')

            # Creating the total hdul
            hdul_new = []
            hdul_new.append(fits.PrimaryHDU(data=new_images[0], header=init_header_SW))
            hdul_new.append(fits.ImageHDU(data=new_images[1], header=init_header_LW))
            hdul_new = fits.HDUList(hdul_new)
            hdul_new.writeto(new_filename, overwrite=True)

            print(f'File nb{loop}, i.e. {filename}, processed.', flush=True)
        print(f'\033[1;33mFor exposure {exposure}, {processed_darks_total_nb} files processed\033[0m')
        if self.processes > 1:
            queue.put((exposure, processed_darks_total_nb, processed_SPIOBSID))
        else:
            return (exposure, processed_darks_total_nb, processed_SPIOBSID)

    def Time_interval(self, filename, files):
        """
        Finding the images in the time interval specified for a given filename. Hence, we are getting the sequences of images needed for each individual
        dark treatment.
        """

        name_dict = common.SpiceUtils.parse_filename(filename)
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
            if interval_filename == filename:
                continue
            name_dict = common.SpiceUtils.parse_filename(interval_filename)
            date = name_dict['time']

            if (date >= date_min) and (date <= date_max):
                interval_filenames.append(interval_filename)
        return interval_filenames

    def mode_along_axis(self, arr):
        return np.bincount(arr.astype('int64')).argmax()
    
    def Mad_mean(self, filenames, detector):
        """
        Function to calculate the mad, mode and mask for a given chunk
        (i.e. spatial chunk with all the temporal values).
        """

        images = np.array([np.array(fits.getdata(common.SpiceUtils.ias_fullpath(filename), detector)[0, :, :, 0], dtype='float64') for filename in filenames]) # TODO: need to check the initial data precision

        # Binning the data
        binned_arr = (images // self.bins) * self.bins

        modes = np.apply_along_axis(self.mode_along_axis, 0, binned_arr).astype('float64')  #a double list comprehension with reshape could be faster
        mads = np.mean(np.abs(images - modes), axis=0).astype('float64')
        return modes, mads 

    def Samedarks(self, filenames):
        # Dictionaries initialisation
        same_darks = {}
        for filename in filenames:
            d = common.SpiceUtils.parse_filename(filename)
            SPIOBSID = d['SPIOBSID']
            if SPIOBSID not in same_darks:
                same_darks[SPIOBSID] = []
            same_darks[SPIOBSID].append(filename)
        return same_darks


if __name__ == '__main__':
    mpl.rcParams['figure.figsize'] = (8, 8)

    import sys

    print(f'Python version is {sys.version}')

    warnings.filterwarnings('ignore', category=mpl.MatplotlibDeprecationWarning)
    test = Cosmicremoval_class(min_filenb=20)
    test.Multiprocess()
    warnings.filterwarnings("default", category=mpl.MatplotlibDeprecationWarning)

