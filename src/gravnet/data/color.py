""""
gravnet"""
import numpy as np
from astropy.visualization import ZScaleInterval, MinMaxInterval, LinearStretch, LogStretch
import matplotlib.pyplot as plt
from PIL import Image
from gravnet.data.fits import FitsData

class ColorData(FitsData):
    """
    A class for managing color image data.

    Attributes:
        tile (str): The tile identifier.
        path (pathlib.Path): The path to the image directory.
        bands (tuple): The bands to use for detection.
        catalog_path (pathlib.Path): The path to the catalog file.
        images_paths (list): The list of paths to the images files."""
    def __init__(self, tile, path = None, bands = ('VIS', 'G', 'R', 'I', 'Z')):
        """
        Initializes an instance of the ColorData class.
        
        Args:
            tile (str): The tile identifier.
            path (pathlib.Path, optional): The path to the image directory. Defaults to None.
            bands (tuple): The bands to use for color data. Defaults to ('VIS', 'G', 'R', 'I', 'Z').
        
        Returns:
            None
        """

        super().__init__(tile, path, bands)
    def compute_color(self, coords, size=(201, 201), stretch = 'minmax,log', save_file = True):
        """
        Computes the color image based on the given coordinates, size, and stretch parameters.

        Args:
            coords (tuple): The coordinates around which to compute the color image.
            size (tuple): The size of the color image. Default is (201, 201).
            stretch (str): The type of stretch to apply to the color image. Default is 'minmax,log'.

        Returns:
            None
        """
        cutouts = self.get_cutouts(coords=coords, size=size, normalize=False)
        stretch = stretch.split(',')
        zscale = ZScaleInterval()
        minmax = MinMaxInterval()
        linear = LinearStretch()
        log = LogStretch()
        for stretch_type in stretch:
            if 'zscale' in stretch_type:
                cutouts = {key: zscale(image) for key, image in cutouts.items()}
            elif 'minmax' in stretch_type:
                cutouts = {key: minmax(image) for key, image in cutouts.items()}
            elif 'linear' in stretch_type:
                cutouts = {key: linear(image) for key, image in cutouts.items()}
            elif 'log' in stretch_type:
                cutouts = {key: log(image) for key, image in cutouts.items()}
        cutouts = {key: (image - np.min(image)) / (np.max(image) - np.min(image)) for
           key, image in cutouts.items()}
        red_channel = (cutouts['R'] * 0.33 + cutouts['I'] * 0.33 + cutouts['Z'] * 0.33) + 1
        green_channel = cutouts['G'] + 1
        blue_channel = cutouts['G'] + 1
        color_image_array = np.stack([red_channel, green_channel, blue_channel], axis=-1)
        colorized_vis_image = cutouts['VIS'][..., np.newaxis] * (color_image_array) - 1
        colorized_vis_image = ((colorized_vis_image - np.min(colorized_vis_image)) /
                                (np.max(colorized_vis_image) - np.min(colorized_vis_image)))
        if save_file:
            Image.fromarray((colorized_vis_image *
                             255).astype(np.uint8)).save(
                f'{coords[0]}_{coords[1]}_color.png', vertical_flip=True)
    
