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
    def __init__(self, processes: int = 15, coefficient: int | float = 6, min_filenb: int = 20, set_min: int = 4,
                time_interval: int = 6, bins: int = 5, stats: bool = True, plots: bool = True):
        
        # Arguments
        self.processes = processes
        self.coef = coefficient
        self.min_filenb = min_filenb
        self.set_min = set_min
        self.stats = stats  # Bool to know if stats will be saved
        self.plots = plots  # boolean to know if plots will be saved
        # self.time_intervals = time_intervals
        self.time_interval = time_interval
        self.bins = bins

        # Code functions
        self.exposures = self.Exposure()

    ################################################ INITIAL functions #################################################
    def Paths(self, time_interval=-1, exposure=-1, detector=-1):
        """
        Function to create all the different paths. Lots of if statements to be able to add files where ever I want
        """

        main_path = os.path.join(os.getcwd(), 'Cosmic_removal_errors3')

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
            print(f'\033[91m ERROR: NO "USABLE" ACQUISITIONS - STOPPING THE RUN\033[0m')
            sys.exit()

    ############################################### CALCULATION functions ##############################################
    @decorators.running_time
    def Multiprocess(self):
        """
        Function for multiprocessing if self.processes > 1. No multiprocessing done otherwise.
        """

        # Choosing to multiprocess or not
        if self.processes > 1:
            print(f'Number of used processes is {self.processes}', flush=True)
            pool = mp.Pool(processes=self.processes)
            pool.starmap(self.Main, self.exposures)
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
        filenames = self.Images_all(exposure)

        if len(filenames) < self.min_filenb:
            print(f'\033[91mInter{self.time_interval}_exp{exposure} -- Less than {self.min_filenb} usable files. '
                  f'Changing exposure times.\033[0m')
            return

        # MULTIPLE DARKS analysis
        same_darks = self.Samedarks(filenames)

        print(f'Inter{self.time_interval}_exp{exposure} -- Starting chunks.')
        for loop, filename in enumerate(filenames):
            if filename in [item for sublist in same_darks.values() for item in sublist if len(sublist) > 3]:
                print(f'\033[31mInter{self.time_interval}_exp{exposure}_imgnb{loop}'
                      f'-- Image from a SPIOBSID set of 3 or more darks. Going to the next acquisition.\033[0m')
                continue

            interval_filenames = self.Time_interval(self.time_interval, filename, filenames)
            
            print(f'Inter{self.time_interval}_exp{exposure}_imgnb{loop}'
                  f' -- Nb of used files: {len(interval_filenames)}')
            
            if len(interval_filenames) < self.set_min:
                print(f'\033[31mInter{self.time_interval}_exp{exposure}_imgnb{loop} '
                      f'-- Less than {self.set_min} files for the processing. Going to the next filename\033[0m')
                continue
            
            new_hdul = []
            treated_pixels = []
            nw_images = []
            for detector in range(2):
                mad, mode = self.Mad_mean(interval_filenames, detector)
                image = np.array(fits.getdata(filename, detector)[0, :, :, 0], dtype='float64')
                mask = image > self.coef * mad + mode

                nw_image = np.copy(image)
                nw_image[mask] = mode[mask]
                nw_images.append(nw_image)

                old_pixels = np.copy(image)
                old_pixels[~mask] = 0
                treated_pixels.append(old_pixels)
            nw_images = np.array(nw_images)
            treated_pixels = np.array(treated_pixels)

            print(f'Treatment for image nb {loop} done. Saving to a fits file.')
            header = fits.getheader(filename, 0)
            nw_header = header.copy()
            nw_header.set('OBS_DESC', value='testing if it works', comment='testing if the comments work')
            init_filename, init_fileextension = os.path.splitext(os.path.basename(filename))

            hdul_new = []
            hdul_new.append(fits.ImageHDU(data=nw_image[0], header=nw_header))
            hdul_new.append(fits.ImageHDU(data=nw_image[1], header=nw_header))
            hdul_new.append(fits.ImageHdu(data=treated_pixels[0], header=nw_header))
            hdul_new.append(fits.ImageHdu(data=treated_pixels[1], header=nw_header))

            nw_filename = f"{init_filename}_1{init_fileextension}"
            hdul_new.writeto(nw_filename, overwrite=True)
            print(f'File nb{loop}, i.e. {os.path.basename(filename)}, processed.', flush=True)

    def Time_interval(self, date_interval, filename, files):
        """
        Finding the images in the time interval specified for a given filename. Hence, we are getting the sequences of images needed for each individual
        dark treatment.
        """

        name_dict = common.SpiceUtils.parse_filename(filename)
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

        interval_filenames = []
        for interval_filename in files:
            if interval_filename == filename:
                continue
            name_dict = common.SpiceUtils.parse_filename(interval_filename)
            date = parse_date(name_dict['time'])

            if (date >= date_min) and (date <= date_max):
                interval_filenames.append(interval_filename)
        return interval_filenames

    def mode_along_axis(self, arr):
        return np.bincount(arr).argmax()
    
    def Mad_mean(self, filenames, detector):
        """
        Function to calculate the mad, mode and mask for a given chunk
        (i.e. spatial chunk with all the temporal values).
        """

        images = np.array([np.array(fits.getdata(filename, detector)[0, :, :, 0], dtype='float64') for filename in filenames]) # TODO: need to check the initial data precision

        # Binning the data
        binned_arr = (images // self.bins) * self.bins

        modes = np.apply_along_axis(self.mode_along_axis, 0, binned_arr).astype('float64')  #a double list comprehension with reshape could be faster
        mads = np.mean(np.abs(images - modes), axis=0).astype('float64')
        return mads, modes # these are all the values for each chunk

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

    print(f'version is {sys.version}')

    warnings.filterwarnings('ignore', category=mpl.MatplotlibDeprecationWarning)
    test = Cosmicremoval_class(min_filenb=20)
    test.Multiprocess()
    warnings.filterwarnings("default", category=mpl.MatplotlibDeprecationWarning)

