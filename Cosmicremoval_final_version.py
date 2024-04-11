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
from multiprocessing import Process
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
    def __init__(self, processes: int = 1, coefficient: int | float = 6, min_filenb: int = 20, set_min: int = 4,
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
            print(f'\033[91m ERROR: NO "USABLE" ACQUISITIONS - STOPPING THE RUN\033[0m')
            sys.exit()
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
            processes = []
            for exposure in self.exposures:
                processes.append(Process(target=self.Main, args=(exposure,)))
            for p in processes:
                p.start()
            for p in processes:
                p.join()
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
                      f'-- Image from a SPIOBSID set of 4 or more darks. Going to the next acquisition.\033[0m')
                continue

            interval_filenames = self.Time_interval(filename, filenames)
            
            print(f'Inter{self.time_interval}_exp{exposure}_imgnb{loop}'
                  f' -- Nb of used files: {len(interval_filenames)}')
            
            if len(interval_filenames) < self.set_min:
                print(f'\033[31mInter{self.time_interval}_exp{exposure}_imgnb{loop} '
                      f'-- Less than {self.set_min} files for the processing. Going to the next filename.\033[0m')
                continue

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
            
            treated_pixels = []
            new_images = []
            for detector in range(2):
                mode, mad = self.Mad_mean(interval_filenames, detector)
                image = np.array(fits.getdata(common.SpiceUtils.ias_fullpath(filename), detector)[0, :, :, 0], dtype='float64')
                mask = image > self.coef * mad + mode
                
                nw_image = np.copy(image)
                nw_image[mask] = mode[mask]
                nw_image = nw_image[np.newaxis, :, :, np.newaxis]
                new_images.append(nw_image.astype('uint16'))

                old_pixels = np.copy(image)
                old_pixels[~mask] = 0
                old_pixels = old_pixels[np.newaxis, :, :, np.newaxis]
                treated_pixels.append(old_pixels.astype('uint16'))
            new_images = np.array(new_images)
            treated_pixels = np.array(treated_pixels)
            print(f'Treatment for image nb {loop} done. Saving to a fits file.')

            # Creating the new_headers
            headers = self.Changing_the_hdu_headers(filename, new_filename, new_images, treated_pixels, new_version)

            # Creating the total hdul
            hdul_new = []
            hdul_new.append(fits.PrimaryHDU(data=new_images[0], header=headers[0][0]))
            hdul_new.append(fits.ImageHDU(data=new_images[1], header=headers[0][1]))
            hdul_new.append(fits.ImageHDU(data=treated_pixels[0], header=headers[1][0]))
            hdul_new.append(fits.ImageHDU(data=treated_pixels[1], header=headers[1][1]))
            hdul_new = fits.HDUList(hdul_new)
            hdul_new.writeto(new_filename, overwrite=True)
            print(f'File nb{loop}, i.e. {filename}, processed.', flush=True)

    def Changing_the_hdu_headers(self, old_filename, new_filename, new_images, new_masks, version):
        """
        Where I put all the changes to the hdu headers for a better organised code.
        """

        # Getting the initial headers
        init_header_SW = fits.getheader(common.SpiceUtils.ias_fullpath(old_filename), 0)
        init_header_LW = fits.getheader(common.SpiceUtils.ias_fullpath(old_filename), 1)

        print(f'the init headers are {init_header_SW}')
        print(f'and {init_header_LW}', flush=True)

        # Creating the new headers
        header_SW_0, header_LW_0 = init_header_SW.copy(), init_header_LW.copy()
        print(f'the copy for the SW gives {header_SW_0}')
        header_SW_0 = self.Header(version, new_filename, header_SW_0, new_images[0], cosmic_extname=header_SW_0['EXTNAME'])
        header_LW_0 = self.Header(version, new_filename, header_LW_0, new_images[1], cosmic_extname=header_LW_0['EXTNAME'])

        header_SW_1, header_LW_1 = header_SW_0.copy(), header_LW_0.copy()
        header_SW_1 = self.Header(version, new_filename, header_SW_1, new_masks[0], cosmic=True, cosmic_extname='Cosmic ray pixels for SW')
        header_LW_1 = self.Header(version, new_filename, header_LW_1, new_masks[0], cosmic=True, cosmic_extname='Cosmic ray pixels for LW')
        return [[header_SW_0, header_LW_0], [header_SW_1, header_LW_1]]
    
    def Header(self, version, new_filename, header, image, cosmic_extname: str, cosmic: bool = False):
        """
        To get the values needed for the headers.
        """

        # Values that are used multiple times or need a long formula
        N = np.count_nonzero(~np.isnan(image))
        image_mean = np.nanmean(image)
        image_rms = np.sqrt(np.nanmean((image - image_mean)**2))
        skewness = np.nansum((image - np.nanmean(image))**3) / ((N - 1) * np.nanstd(image)**3)
        kurtosis = N * np.nansum((image - np.nanmean(image))**4) / np.nansum((image - np.nanmean(image)**2)**2)

        # The dictionary with all the header statistics
        headers_dict = {
            'EXTNAME': [cosmic_extname, 'Extension name'],
            'FILENAME': [new_filename, 'Filename'],
            'NWIN': [4, 'Total number of windows in this file'],
            'NWIN_PRF': [4, 'Number of windows not Dumbbell or Intensity'],
            'VERSION': [f'{version:02d}      ', 'Incremental file version number'],
            'DATAMIN': [np.nanmin(image),     '[adu] Minimum data value'],
            'DATAMAX': [np.nanmax(image),     '[adu] Maximum data value'],
            'DATAMEAN': [image_mean,          '[adu] Mean    data value'],
            'DATAMEDN': [np.nanmedian(image), '[adu] Median  data value'],
            'DATAP01': [np.nanpercentile(image, 1),  '[adu] 1st  percentile of data values'],
            'DATAP10': [np.nanpercentile(image, 10), '[adu] 10th percentile of data values'],
            'DATAP25': [np.nanpercentile(image, 25), '[adu] 25th percentile of data values'],
            'DATAP75': [np.nanpercentile(image, 75), '[adu] 75th percentile of data values'],
            'DATAP90': [np.nanpercentile(image, 90), '[adu] 90th percentile of data values'],
            'DATAP95': [np.nanpercentile(image, 95), '[adu] 95th percentile of data values'],
            'DATAP98': [np.nanpercentile(image, 98), '[adu] 98th percentile of data values'],
            'DATAP99': [np.nanpercentile(image, 99), '[adu] 99th percentile of data values'],
            'DATARMS': [image_rms, '[adu] RMS dev: sqrt(sum((data-DATAMEAN)^2)/N)'],
            'DATANRMS': [image_rms / image_mean, 'Normalised RMS dev: DATARMS/DATAMEAN'],
            'DATAMAD': [np.nanmean(np.abs(image - image_mean)), '[adu] Mean abs dev: sum(abs(data-DATAMEAN))/N '],
            'DATASKEW': [skewness, 'Data skewness'],
            'DATAKURT': [kurtosis, 'Data kurtosis'],
        }

        if cosmic:
            headers_dict['EXTNAME'] = [cosmic_extname, 'Extension name']

        headers_string_n_key = self.Header_string(headers_dict)
        new_header = self.Getting_the_new_header(headers_string_n_key, header._cards)

        # Setting the forced comment changes from astropy back to the initially used one
        # keys_changed = [
        #     'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'NAXIS4', 'EXTEND', 'BSCALE', 'BZERO',
        # ]
        # for key in keys_changed:
        #     new_header[key].comments = header[key].comments
        return new_header
    
    def Getting_the_new_header(self, headers_string_n_key, cards):
        """
        To find the right header card and swap it with the new value.
        """
        print(f'The corresponding cards for the header copy are {cards}', flush=True)

        for (key, string) in headers_string_n_key:
            for loop, card in enumerate(cards):
                if card.keyword == key:
                    break
            del cards[loop]
            cards.insert(loop, fits.Card.fromstring(string))
        return fits.Header(cards=cards)

    def Header_string(self, header_dict: dict) -> list:
        """
        Given that in the initial hdul headers have all the same lengths till the header, I need a custom function to
        have the same formatting.
        """

        header_strings_n_key = []
        for key, (value, comment) in header_dict.items():
            value_length = 7 if key not in ['DATARMS', 'DATANRMS', 'DATAMAD', 'DATASKEW', 'DATAKURT'] else 13
            if not isinstance(value, str):
                value_string = self.Header_value_length(value, value_length)
                header_strings_n_key.append(
                    (key, f'{self.Format_string_left(key, self.header_key_length)}={self.Format_string_right(value_string, self.header_value_length)} / {comment}'[:80])
                )
            else:
                value_string = f" '{value}'"
                header_strings_n_key.append(
                    (key, f'{self.Format_string_left(key, self.header_key_length)}={self.Format_string_left(value_string, self.header_value_length)} / {comment}'[:80])
                )
        return header_strings_n_key    
        
    def Header_value_length(self, value: int | float, length: int) -> str:
        """
        Changing the value to a string of a given size (i.e. adding decimals or taking some away).
        """

        a = 0
        if value < 0:
            a += 1 if len(str(int(value))) != 1 else 0
        int_length = len(str(int(value)))
        number = a + length - (int_length + 1)
        return format(value, f'.{number}f') if number > 0 else str(int(value)) + '.' if number == 0 else str(int(value))  
    
    def Format_string_right(self, string: str, length: int) -> str:
        """
        Creating a string with a set length and starting from the right.
        """

        return string.rjust(length)
    
    def Format_string_left(self, string: str, length: int) -> str:
        """
        Creating a string with a set length and starting from the left.
        """

        return string.ljust(length)

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

    print(f'version is {sys.version}')

    warnings.filterwarnings('ignore', category=mpl.MatplotlibDeprecationWarning)
    test = Cosmicremoval_class(min_filenb=20)
    test.Multiprocess()
    warnings.filterwarnings("default", category=mpl.MatplotlibDeprecationWarning)

