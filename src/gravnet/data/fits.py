"""
gravnet"""
import os
import pathlib
import re
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from astropy.table import Table
import tensorflow as tf
import numpy as np
from gravnet.constants import CATALOG_PATTERN
from gravnet.constants import IMAGE_PATTERN_G
from gravnet.constants import IMAGE_PATTERN_R
from gravnet.constants import IMAGE_PATTERN_I
from gravnet.constants import IMAGE_PATTERN_Z
from gravnet.constants import IMAGE_PATTERN_VIS
from gravnet.constants import IMAGE_PATTERN_NISP_H
from gravnet.constants import IMAGE_PATTERN_NISP_J
from gravnet.constants import IMAGE_PATTERN_NISP_Y

class FitsData():
    """
    A class for managing FITS image data.

    Attributes:
        tile (str): The tile identifier.
        path (pathlib.Path): The path to the image directory.
        bands (tuple): The bands to use for detection.
        catalog_path (pathlib.Path): The path to the catalog file.
        images_paths (list): The list of paths to the images files."""
    def __init__(self, tile, path = None, bands = ('VIS', 'H', 'J', 'Y')):
        """
        Initializes an instance of the Image class.

        Args:
            tile (str): The tile identifier.
            path (str, optional): The path to the image directory. Defaults to None.
            bands (tuple): The bands to use for detection. Can be 'VIS', 'H', 'J', 'Y',
            G, R, I, or Z. Defaults to ('VIS', 'H', 'J', 'Y').

        Attributes:
            tile (str): The tile identifier.
            path (pathlib.Path): The path to the image directory.
            catalog (pathlib.PurePath): The path to the catalog file.
            images (pathlib.PurePath): The path to the images file.
            objects (dict): The dictionary of objects.

        Returns:
            None
        """
        self.tile = str(tile) if tile is not None else tile
        self.path = pathlib.Path(path)
        self.bands = bands
        self.catalog_paths = self.get_catalog_file()
        self.images_paths = self.get_images_files()
        self.objects = self.get_objects()

    def get_catalog_file(self):
        """
        Finds and returns the name of the catalog file in the current
        directory that matches the given tile identifier.

        Returns:
            str: The name of the catalog file if found, None otherwise.
        """
        pattern = re.compile('.*'+CATALOG_PATTERN+self.tile+'.*.fits')
        catalog_list = []
        for filename in os.listdir(pathlib.PurePath(self.path, 'CAT')):
            if pattern.match(filename):
                catalog_list.append(pathlib.PurePath(self.path, 'CAT', filename))
        if len(catalog_list) == 0:
            print('No catalog file found')
            return None
        return catalog_list

    def get_images_files(self):
        """
        Finds and returns the name of the images file in the current directory
        that matches the given tile identifier.

        Returns:
            str: The name of the images file if found, None otherwise.
        """
        pattern = {
            'VIS': re.compile('.*'+IMAGE_PATTERN_VIS+self.tile+'.*.fits'),
            'H': re.compile('.*'+IMAGE_PATTERN_NISP_H+self.tile+'.*.fits'),
            'J': re.compile('.*'+IMAGE_PATTERN_NISP_J+self.tile+'.*.fits'),
            'Y': re.compile('.*'+IMAGE_PATTERN_NISP_Y+self.tile+'.*.fits'),
            'G': re.compile('.*'+IMAGE_PATTERN_G+self.tile+'.*.fits'),
            'R': re.compile('.*'+IMAGE_PATTERN_R+self.tile+'.*.fits'),
            'I': re.compile('.*'+IMAGE_PATTERN_I+self.tile+'.*.fits'),
            'Z': re.compile('.*'+IMAGE_PATTERN_Z+self.tile+'.*.fits')
        }
        files_path = {
            'VIS': '',
            'H': '',
            'J': '',
            'Y': '',
            'G': '',
            'R': '',
            'I': '',
            'Z': ''
        }
        for band in self.bands:
            for filename in os.listdir(pathlib.PurePath(self.path, band)):
                if pattern[band].match(filename):
                    files_path[band] = self.path.joinpath(band, filename)
        for band, path in files_path.items():
            if path == '' and band in self.bands:
                print(f'No file found in band {band} for tile: {self.tile}')
        return files_path

    def get_objects(self):
        """
        Retrieves the objects from the catalog file and returns them as a dictionary.

        Returns:
            dict: A dictionary containing the objects from the catalog file.
            Each object is represented by a key-value pair, where the key is the 'OBJECT_ID'
            and the value is a dictionary containing the following information:
                - 'ra' (float): The right ascension of the object.
                - 'dec' (float): The declination of the object.
                - 'kron_rad' (float): The Kron radius of the object.
                - 'mag' (float): The magnitude of the object.
        """
        objects_dict = {}
        for catalog_path in self.catalog_paths:
            catalog = fits.open(catalog_path)
            catalog_data = catalog[1].data
            catalog_table = Table(catalog_data)
            for line in catalog_table:
                objects_dict[line['OBJECT_ID']] = {'ra': line['RIGHT_ASCENSION'],
                                                'dec': line['DECLINATION'],
                                                'kron_rad': line['KRON_RADIUS'],
                                                'mag': line['MAG_STARGAL_SEP'],
                                                'flag': line['FLAG_VIS']}
        return objects_dict

    def get_cutouts(self, coords=(60, 60), size=(201, 201), normalize=True):
        """
        Retrieves cutouts of the image data for specified bands and coordinates, and normalizes the data.

        Args:
            coords (tuple): The coordinates to center the cutout on. Defaults to (60, 60).
            size (tuple): The size of the cutout. Defaults to (201, 201).
            normalize (bool): Flag to normalize the data. Defaults to True.

        Returns:
            dict: A dictionary containing the cutout data for each band specified in `self.bands`.
                The keys are the band names, and the values are the corresponding cutout data as numpy arrays.
        """
        tensor_dict = {}
        for band in self.bands:
            image = fits.open(self.images_paths[band])
            data = image[0].data
            center_sky = SkyCoord(ra=coords[0],
                                dec=coords[1],
                                unit='deg')
            wcs = WCS(image[0].header)
            x_center, y_center = wcs.all_world2pix(center_sky.ra.deg, center_sky.dec.deg, 0)
            center = (x_center, y_center)
            size = u.Quantity(size, u.pix)
            cutout = Cutout2D(data, center, size, wcs=wcs)
            data = cutout.data
            if normalize:
                data = (data - np.min(data)) / (np.max(data) - np.min(data))
            tensor_dict[band] = data
            image.close()

        return tensor_dict

    def get_tf_tensor(self, object_id, size=(201, 201)):
        """
        Get a TensorFlow tensor for the given object ID.

        Args:
            object_id (str): The identifier of the object.
            size (tuple, optional): The size of the tensor. Defaults to (201, 201).

        Returns:
            tf.Tensor: The TensorFlow tensor.

        This function retrieves the coordinates of the object and calls the `get_cutouts`
        method to get the cutouts of the image data.
        It then calls the `concatenate_bands` method to concatenate the cutouts into a tensor.
        If the tensor has more than 8 channels, it expands the tensor along the last axis.
        Finally, it expands the tensor along the first axis and returns it.
        """
        coords = [self.objects[object_id]['ra'], self.objects[object_id]['dec']]
        tensor_dict = self.get_cutouts(coords=coords, size=size)
        tensor = self.concatenate_bands(tensor_dict)
        if tensor.shape[-1] > 8:
            tensor = np.expand_dims(tensor, axis=-1)
        tensor = tf.expand_dims(tensor, axis=0)
        return tensor

    def concatenate_bands(self, tensor_dict):
        """
        Concatenates the bands in the tensor dictionary along the third axis.

        Args:
            tensor_dict (dict): A dictionary containing the bands as keys and the data as values.

        Returns:
            numpy.ndarray: The concatenated tensor representing the bands.
        """
        tensor = tensor_dict[self.bands[0]]
        for band in self.bands[1:]:
            np.concatenate((tensor, tensor_dict[band]), axis=2)
        return tensor

    def save_fits(self, image, path, header=None):
        """
        Save the given image as a FITS file.

        Args:
            image: The image to be saved.
            path: The path to save the image to.

        Description:
        """
        hdu = fits.PrimaryHDU(image, header=header)

        hdu.writeto(path, overwrite=True)

    def draw_boxes(self, coords, box_size=200, thickness=1, value=1):
        """
        Draw boxes on an image based on the given coordinates and save the modified 
        image as a FITS file.

        Args:
            path (str): The path to save the modified image.
            coords (List[Tuple[float, float]]): A list of coordinates (ra, dec) in degrees.
            box_size (int, optional): The size of the boxes in pixels. Defaults to 200.
            thickness (int, optional): The thickness of the box lines. Defaults to 1.
            value (int, optional): The value to fill the box lines with. Defaults to 1.

        Returns:
            None
        """
        image = fits.open(self.images_paths['VIS'])
        data = image[0].data
        wcs = WCS(image[0].header)
        for coord in coords:
            center_sky = SkyCoord(ra=coord[0], dec=coord[1], unit='deg')
            x_center, y_center = wcs.all_world2pix(center_sky.ra.deg, center_sky.dec.deg, 0)
            x_min = int(x_center - box_size // 2)
            x_max = int(x_center + box_size // 2)
            y_min = int(y_center - box_size // 2)
            y_max = int(y_center + box_size // 2)

            # Dessiner les lignes du cadre
            data[y_min:y_min+thickness, x_min:x_max+1] = value # Ligne supérieure
            data[y_max-thickness+1:y_max+1, x_min:x_max+1] = value  # Ligne inférieure
            data[y_min:y_max+1, x_min:x_min+thickness] = value  # Ligne gauche
            data[y_min:y_max+1, x_max-thickness+1:x_max+1] = value  # Ligne droite

        self.save_fits(data, 'lenses_' + self.tile + '.fits', image[0].header)
