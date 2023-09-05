"""
This code was created to take away the cosmic ray hits on darks.
It generates a histogram from a given .fits dark image to use the highest occurence intensity value of the histogram as
a reference to quantify the variability of the intensity values. This variability (mad) is defined as the absolute
deviation with respect to the reference value. From there, the code "mad clips" the dark to create a mask. This mask is
then filtered by taking away all mask regions that are less than 2 connected pixels. Lastly, from the given filtered
mask, a new mask is created by region growth/flooding the corresponding pixels of the filtered mask. The flooding
tolerance is also proportionate to the mad value. The mask created represents the positions of the pixels that are
flagged as cosmic rays.

Creation date: 2023.05.12  (y.m.d)
"""

# Needed libraries
import os
import common
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from astropy.io import fits
from simple_decorators import decorators
from skimage.segmentation import flood
import multiprocessing as mp
from collections import Counter
from dateutil.parser import parse as parse_date
from skimage.measure import label, regionprops

class Cosmicremoval_class:
    # Finding the L1 darks
    cat = common.SpiceUtils.read_spice_uio_catalog()
    filters = cat.STUDYDES.str.contains('dark') & (cat['LEVEL'] == 'L1')
    res = cat[filters]

    def __init__(self, mad_clip, noise_limit, processes, min_filenb=40):
        # For the prints
        self.process_id = '000'
        self.dpi = 1024 / 8

        # Inputs
        self.min_filenb = min_filenb
        self.mad_clip = mad_clip
        self.noise_limit = noise_limit
        self.processes = processes

        # Code functions
        self.exposures = self.Exposure()

    ############################################### STRUCTURE functions ################################################
    def Paths(self, exposure='none', detector='none'):
        """Function to create all the different paths. Lots of if statements to be able to add files where ever I want
        """

        main_path = os.path.join(os.getcwd(), f'vf_Spatial_mad{self.mad_clip}_noise{self.noise_limit}')

        if exposure != 'none':
            exposure_path = os.path.join(main_path, f'Exposure{exposure}')

            if detector != 'none':
                detector_path = os.path.join(exposure_path, f'Detector{detector}')
                # Main paths
                initial_paths = {'Main': main_path, 'Exposure': exposure_path, 'Detector': detector_path}
                # Secondary paths
                directories = ['Darks', 'Masks', 'Same', 'Histograms', 'Same_errors', 'Statistics']
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
        print(f'\033[33mExposure times with less than \033[1m{self.min_filenb}\033[0m\033[33m darks are not kept')
        print(f'Exposure times kept are {exposure_used}\033[0m')

        exposure_used = np.array(exposure_used)  # TODO: not sure if I need this line

        return exposure_used

    def Images_all(self, exposure, detector):
        """Function to get, for a certain exposure time and detector nb, the corresponding images, distance to sun,
        temperature array and the corresponding filenames"""

        # Filtering the data by exposure time
        filter = Cosmicremoval_class.res.XPOSURE == exposure
        res = Cosmicremoval_class.res[filter]

        # Important stats
        filenames = np.array(list(res['FILENAME']))

        # Variable initialisation
        a = 0
        images_all = []
        filenames_nw = []
        for file in filenames:
            # Opening the files
            hdul = fits.open(common.SpiceUtils.ias_fullpath(file))
            all_data = hdul[detector].data
            image = np.double(np.array(all_data[0, :, :, 0]))

            # Temperature check
            if detector == 0:
                temp = hdul[0].header['T_SW']
            else:
                temp = hdul[0].header['T_LW']
            if temp > 0:
                a += 1
                continue

            # Saving the data
            images_all.append(image)
            filenames_nw.append(file)
        # Changing to arrays
        images_all = np.array(images_all)
        filenames_nw = np.array(filenames_nw)

        print(f'Exp{exposure}_det{detector} -- Nb files with high temp: {a}')
        return images_all, filenames_nw

    def Bins(self, data):
        """Small function to calculate the appropriate bin count"""

        bins = int(data.size / 3000)
        if bins < 20:
            bins = 20

        return bins

    def Contours(self, mask):
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

    def Multi(self):
        totstd_percentage = []
        totmad_percentage = []

        if self.processes > 1:
            pool = mp.Pool(processes=self.processes)
            args = [(exposure, totmad_percentage, totstd_percentage) for exposure in self.exposures]
            pool.starmap(self.Main, args)
            pool.close()
            pool.join()

        else:
            for exposure in self.exposures:
                totmad_percentage, totstd_percentage = self.Main(exposure, totmad_percentage, totstd_percentage)
            totmad_percentage = np.array(totmad_percentage)
            totstd_percentage = np.array(totstd_percentage)

            print(f'The mad_percentage std is {round(np.std(totmad_percentage), 2)}. '
                  f'The std_percentage std is{round(np.std(totstd_percentage), 2)}.')

    @decorators.running_time
    def Main(self, exposure, totmad_percentage, totstd_percentage):
        # MAIN LOOP to multiprocess
        data_pandas_exposure = pd.DataFrame()
        for detector in range(2):
            # Initialisation of the code
            paths = self.Paths(exposure=exposure, detector=detector)
            print(f"Exp{exposure}_det{detector} -- Paths finished")
            images, filenames = self.Images_all(exposure, detector)
            print(f'Exp{exposure}_det{detector} -- Acquisitions downloaded.')
            same_darks, positions = self.Samedarks(filenames)
            data_pandas_detector = pd.DataFrame()
            for SPIOBSID, files in same_darks.items():
                if len(files) < 3:
                    continue
                std_percentage = []  # only for the given SPIOBSID
                mad_percentage = []
                print(f'Exp{exposure}_det{detector}_ID{SPIOBSID} -- Filenames are: {files}')

                loops = positions[SPIOBSID]
                data = images[loops]
                masks = []
                modes = []
                mads = []
                for loop, image in enumerate(data):
                    # Initialisation for the filename dictionary
                    name_dict = common.SpiceUtils.parse_filename(files[loop])

                    # Getting the mad mode values
                    mad, mode, bin_edges, hist = self.mad_mode_func(image)
                    std_per, mad_per = self.Percentages_stdnmad(image, mad, mode)

                    image = image - mode
                    # Creation and labeling of the first mask
                    first_mask = self.madclip_func(image, mad)
                    label_firstmask = label(first_mask)

                    # Creation and labeling of the second ("filtered") mask
                    second_mask = self.maskneighbours_func(image, label_firstmask)
                    label_secondmask = label(second_mask)

                    # Creation of the final mask
                    mask_tot = self.finalmask_func(image, label_secondmask, mad)

                    # Saving the stats
                    masks.append(mask_tot)
                    modes.append(mode)
                    mads.append(mad)
                    # TOT std and mad stats for later analysis
                    std_percentage.append(std_per)
                    mad_percentage.append(mad_per)

                    # Plotting
                    self.Histo_plotting(image, paths, name_dict, loop, mad, mode, bin_edges, hist, std_per, mad_per)
                print(f'Exp{exposure}_det{detector}_ID{SPIOBSID} -- Mad calculations+flooding finished.')
        #
        #         masks = np.array(masks)
        #         modes = np.array(modes)
        #         totstd_percentage.extend(std_percentage)
        #         totmad_percentage.extend(mad_percentage)
        #
        #         # Stats calculations
        #         nw_masks = self.Stats(data, masks, modes)
        #
        #         # MULTIPLE DARKS plotting
        #         # self.Medianplotting(paths, files, data, masks, modes)
        #         # self.Stats_plotting(paths, files, data, nw_masks, modes, SPIOBSID)
        #
        #         # Saving the stats in a csv file
        #         data_pandas = self.Unique_datadict(exposure, detector, files, mads, modes, std_percentage,
        #                                            mad_percentage, masks, nw_masks)
        #         data_pandas_detector = pd.concat([data_pandas_detector, data_pandas])
        #         csv_name = f'Info_for_ID{SPIOBSID}.csv'
        #         data_pandas.to_csv(os.path.join(paths['Statistics'], csv_name), index=False)
        #     # Combining the dictionaries
        #     data_pandas_exposure = pd.concat([data_pandas_exposure, data_pandas_detector])
        # # Saving a csv file for each exposure time
        # paths = self.Paths(exposure=exposure)
        # csv_name = f'Info_for_exp{exposure}.csv'
        # data_pandas_exposure.to_csv(os.path.join(paths['Exposure'], csv_name), index=False)
        # print(f'Exp{exposure} -- CSV file created')

        return totmad_percentage, totstd_percentage

    def Unique_datadict(self, exposure, detector, files, mads, modes, std_percentage, mad_percentage, masks, nw_masks):
        """Function to create a dictionary containing some useful information on each exposure times. This is
         done to save the stats in a csv file when the code finishes running."""

        # Initialisation
        name_dict = common.SpiceUtils.parse_filename(files[0])
        date = parse_date(name_dict['time'])

        # Creation of the stats
        tot_detection = np.sum(masks)
        tot_error = np.sum(nw_masks)
        group_nb = len(files)
        group_date = f'{date.year:04d}{date.month:02d}{date.day:02d}'
        SPIOBSID = name_dict['SPIOBSID']
        # Corresponding lists or arrays
        a, b, c, d, e, f, g = np.full((group_nb, 7),
                                      [exposure, detector, group_date, group_nb, tot_detection, tot_error, SPIOBSID]).T
        detections = np.sum(masks, axis=(1, 2))
        errors = np.sum(nw_masks, axis=(1, 2))

        # Special cases
        if tot_error == 0:
            tot_ratio = np.full(group_nb, np.nan)
        else:
            ratios = tot_error / tot_detection
            tot_ratio = np.full(group_nb, ratios)
        ratio = []
        for loop, val in enumerate(errors):
            if val == 0:
                ratio_val = np.nan
            else:
                ratio_val = val / detections[loop]
            ratio.append(ratio_val)

        data_dict = {'Exposure time': a, 'Detector': b, 'Group date': c, ' Nb of files in group': d,
                     'Tot nb of detections': e, 'Tot nb of errors': f, 'Ratio errors/detections': tot_ratio,
                     'Filename': files, 'SPIOBSID': g, 'Mode': modes, 'Mode absolute deviation': mads,
                     'Nb of detections': detections, 'Nb of errors': errors, 'Ratio': ratio,
                     'mode +/- std = {}% data': std_percentage, 'mode +/- mad = {}% data': mad_percentage}
        # TODO: I need to add another dict with the error ratios by detector and exposure

        Pandasdata = pd.DataFrame(data_dict)
        return Pandasdata

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

    def Medianplotting(self, paths, files, data, masks, modes):
        data_med = np.median(data, axis=0)

        for loop, image in enumerate(data):
            file = files[loop]
            mask = masks[loop]
            mode = modes[loop]
            lines = self.Contours(mask)
            name_dict = common.SpiceUtils.parse_filename(file)
            date = parse_date(name_dict['time'])
            # Calculate the first and last percentiles of the image data
            first_percentile = np.percentile(image, 1)
            last_percentile = np.percentile(image, 99.99)
            if first_percentile < 100:
                first_percentile = 100

            # DARKS plotting
            plot_name = f'Log_dark_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            img = plt.imshow(image, interpolation='none')
            plt.title(f'Dark: {file}')
            # Create a logarithmic colormap
            log_cmap = mcolors.LogNorm(vmin=first_percentile, vmax=last_percentile)
            img.set_norm(log_cmap)
            cbar = plt.colorbar(img)
            cbar.locator = ticker.MaxNLocator(nbins=5)  # Adjust the number of ticks as desired
            cbar.update_ticks()
            plt.savefig(os.path.join(paths['Darks'], plot_name), dpi=self.dpi)
            plot_name = f'Log_darkcont_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            for line in lines:
                plt.plot(line[1], line[0], color='r', linewidth=0.05)
            plt.savefig(os.path.join(paths['Darks'], plot_name), dpi=self.dpi)
            plt.close()

            # DIF MEDIAN plotting
            med_dif = image - data_med
            plot_name = f"Dif_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf"
            plt.imshow(med_dif, interpolation='none', vmin=-100, vmax=400)
            plt.title(f"Dif with median: {file}")
            plt.colorbar()
            plt.savefig(os.path.join(paths['Same'], plot_name), dpi=self.dpi)
            plot_name = f"Dif_cont_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf"
            for line in lines:
                plt.plot(line[1], line[0], color='r', linewidth=0.05)
            plt.savefig(os.path.join(paths['Same'], plot_name), dpi=self.dpi)
            plt.close()

            # Mask plotting
            plot_name = f"Mask_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf"
            plt.imshow(mask, interpolation='none')
            plt.title(f'Final: {file}')
            plt.colorbar()
            plt.savefig(os.path.join(paths['Masks'], plot_name), dpi=self.dpi)
            plt.close()

            # FINAL plotting
            img = np.copy(image)
            plot_name = f"Final_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf"
            img[mask] = mode
            plt.imshow(img, interpolation='none')
            plt.title(f'Final: {file}')
            plt.colorbar()
            plt.savefig(os.path.join(paths['Same'], plot_name), dpi=self.dpi)
            plot_name = f"Final_cont_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf"
            for line in lines:
                plt.plot(line[1], line[0], color='r', linewidth=0.05)
            plt.savefig(os.path.join(paths['Same'], plot_name), dpi=self.dpi)
            plt.close()

            # FINAL DIF MEDIAN plotting
            med_difnw = img - data_med
            plot_name = f"Dif_nw_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf"
            plt.imshow(med_difnw, interpolation='none', vmin=-100, vmax=400)
            plt.title(f"Dif with median: {file}")
            plt.colorbar()
            plt.savefig(os.path.join(paths['Same'], plot_name))
            plot_name = f"Dif_contnw_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf"
            for line in lines:
                plt.plot(line[1], line[0], color='r', linewidth=0.05)
            plt.savefig(os.path.join(paths['Same'], plot_name))
            plt.close()

    def mad_mode_func(self, data):
        """
        Function to quantify the variability of the intensities. It finds the value with the highest occurence to
        then be used as a reference to quantify the variability of the data.
        :return: mad: absolute deviation with respect to the reference value
                 max_intensity_value: the intensity value of the reference
        """
        bins = self.Bins(data)

        # Creating the histogram and getting setting the max occurrence intensity to be the reference
        hist, bin_edges = np.histogram(data, bins=bins)
        max_bin_index = np.argmax(hist)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        max_intensity_value = bin_centers[max_bin_index]

        # Determination of the mode absolute deviation
        mad = np.mean(np.abs(data - max_intensity_value))

        return mad, max_intensity_value, bin_edges, hist

    def Percentages_stdnmad(self, data, mad, mode):

        std_filter = (data > mode - 1 * np.std(data)) & (data < mode + np.std(data))
        mad_filter = (data > mode - 1 * mad) & (data < mode + mad)
        std_kept = data[std_filter]
        mad_kept = data[mad_filter]
        std_percentage = std_kept.size / data.size * 100
        mad_percentage = mad_kept.size / data.size * 100

        return std_percentage, mad_percentage

    def Histo_plotting(self, data, paths, name_dict, loop, mad, mode, bin_edges, hist, std_per, mad_per):
        date = parse_date(name_dict['time'])

        # REF HISTO plotting
        hist_name = f'histogram_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.png'
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
        plt.axvline(mode + np.std(data), color='red', linestyle='--', label='2 * standard deviation, '
                                                                            f'i.e. {round(std_per, 2)}%')
        plt.axvline(mode - np.std(data), color='red', linestyle='--')
        plt.axvline(mode + mad, color='black', linestyle='-', label=f'2 * m.a.d., i.e. {round(mad_per, 2)}%')
        plt.axvline(mode - mad, color='black', linestyle='-')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'mode: {round(mode, 2)}; mad: {round(mad, 2)}; std: {round(np.std(data), 2)}.', fontsize=12)
        plt.xlabel('Detector count', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig(os.path.join(paths['Histograms'], hist_name))
        plt.close()

    def madclip_func(self, image, mad):
        """
        Function to create the first mask by mad clipping the initial image.
        :return: mask: mask with the cosmics with some "noise" flagged as True (==1)
        """

        # Creation of the mask representing mad clipping
        mask = np.zeros_like(image, dtype='bool')
        mask[image > self.mad_clip * mad] = True

        return mask

    def maskneighbours_func(self, image, label):
        """
        Function to filter the initial mask so that only regions (defined with 4 neighbours) with more than 2 pixels
        are kept.
        :param mask:
        :return:
        """
        neighbour_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        processed = set()
        new_mask = np.zeros_like(image, dtype='bool')

        for region in regionprops(label_image=label):
            if region.num_pixels < 3 or region.num_pixels == 1048576:
                continue
            full_pixels = set()  # set with pixel positions
            coords = region.coords
            coords = np.matrix.tolist(coords)

            for [r, c] in coords:
                for (dr, dc) in neighbour_directions:
                    r_neighbour, c_neighbour = r + dr, c + dc
                    if [r_neighbour, c_neighbour] in coords:
                        full_pixels.add((r_neighbour, c_neighbour))

            if len(full_pixels) > 2:
                processed.update(full_pixels)  # saving the positions if more than 2 adjacent pixels
        for (r, c) in processed:
            new_mask[r, c] = True

        return new_mask

    def finalmask_func(self, image, label_mask, mad):
        """
        Function to create the final mask given the labeling of the "filtered" mask.
        This final mask is created by region growth/flooding the pixels of the "filtered" mask.
        The flood parameters are calculated using the seed point, the mad value and a constant multiplying
        the mad value. This is done to be able to try and separate the signal from the cosmic ray hit or the ones
        coming from the thermal electrons.
        :param label_mask: the label of the regions of the "filtered" mask
        :return: sends the mask values directly to the cosmic_removal.plotting_func()
        """

        # Creating an initially empty total mask
        mask_tot = np.zeros_like(image, dtype=bool)

        # Selecting each mask regions
        for region in regionprops(label_image=label_mask, intensity_image=image):
            # Finding the pixel with the highest intensity to use as a seed for region flooding
            intensity = region.image_intensity
            max_coords = np.argwhere(intensity == intensity.max())
            max_coords = (max_coords[0, 0] + region.bbox[0], max_coords[0, 1] + region.bbox[1])

            # Flood the region and get the corresponding mask.
            tolerance = image[max_coords[0], max_coords[1]] - self.noise_limit * mad
            if tolerance < 0:  # the tolerance cannot be negative, hence this if statement.
                continue
            flooded_region = flood(image, (max_coords[0], max_coords[1]), tolerance=tolerance)

            # Adding the created region specific mask to the total mask
            mask_tot[flooded_region] = True

        return mask_tot

    def Stats(self, data, masks, modes):
        """Function to calculate some stats to have an idea of the efficacy of the method. The output is a set of masks
        giving the positions where the method outputted a worst result than the initial image"""

        # Initialisation
        nw_data = np.copy(data)
        data_med = np.median(data, axis=0)
        meds_dif = data - data_med

        # Difference between the end result and the initial one
        for loop in range(len(modes)):
            nw_data[loop, masks[loop]] = modes[loop]
        nw_meds_dif = nw_data - data_med

        # Creating a new set of masks that shows where the method made an error
        nw_masks = np.zeros_like(masks, dtype='bool')
        filters = abs(nw_meds_dif) > abs(meds_dif)
        nw_masks[filters] = True

        return nw_masks

    def Stats_plotting(self, paths, files, data, nw_masks, modes, SPIOBSID):
        sum_masks = np.sum(nw_masks, axis=0)
        data_med = np.median(data, axis=0)
        file = files[0]
        name_dict = common.SpiceUtils.parse_filename(file)
        date = parse_date(name_dict['time'])

        # Total error mask
        masks_name = f'Masks_errors_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}.pdf'
        plt.imshow(sum_masks, interpolation='none')
        plt.title(f'All masks errors: {SPIOBSID}')
        plt.colorbar()
        plt.savefig(os.path.join(paths['Same_errors'], masks_name))
        plt.close()

        for loop, image in enumerate(data):
            file = files[loop]
            nw_mask = nw_masks[loop]
            mode = modes[loop]
            name_dict = common.SpiceUtils.parse_filename(file)
            date = parse_date(name_dict['time'])
            lines = self.Contours(nw_masks[loop])

            # DARK ERROR plotting
            dark_name = f'Dark_errors_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            plt.imshow(image, interpolation='none')
            plt.title(f'Dark: {file}')
            plt.colorbar()
            for line in lines:
                plt.plot(line[1], line[0], color='g', linewidth=0.05)
            plt.savefig(os.path.join(paths['Same_errors'], dark_name))
            plt.close()

            # MASK ERROR plotting
            mask_name = f'Mask_errors_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            plt.imshow(nw_mask, interpolation='none')
            plt.title(f'Errors: {file}')
            plt.savefig(os.path.join(paths['Same_errors'], mask_name))
            plt.close()

            # Dif plotting with errors
            med_dif = image - data_med
            plot_name = f'Dif_errors_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            plt.imshow(med_dif, interpolation='none', vmin=-100, vmax=400)
            for line in lines:
                plt.plot(line[1], line[0], color='g', linewidth=0.05)
            plt.colorbar()
            plt.savefig(os.path.join(paths['Same_errors'], plot_name))
            plt.close()

            # Final result plotting with errors
            img = np.copy(image)
            img[nw_mask] = mode
            med_difnw = img - data_med
            plot_name = f'Final_conterrors_{date.year:04d}{date.month:02d}{date.day:02d}_H{date.hour:02d}_{loop:02d}.pdf'
            plt.imshow(med_difnw, interpolation='none', vmin=-100, vmax=400)
            for line in lines:
                plt.plot(line[1], line[0], color='g', linewidth=0.05)
            plt.colorbar()
            plt.savefig(os.path.join(paths['Same_errors'], plot_name))
            plt.close()




# Need to finish the last images
if __name__ == '__main__':
    mpl.rcParams['figure.figsize'] = (8, 8)
    warnings.filterwarnings('ignore', category=mpl.MatplotlibDeprecationWarning)
    test = Cosmicremoval_class(mad_clip=13, noise_limit=8, processes=1)
    test.Multi()
    warnings.filterwarnings("default", category=mpl.MatplotlibDeprecationWarning)
