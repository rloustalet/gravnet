"""
gravnet"""
import pathlib
import logging
import numpy as np
import yaml
from rich.progress import track
from tensorflow.keras.models import load_model
from gravnet.data.fits import FitsData

logging.getLogger("tensorflow").setLevel(logging.WARNING)
class Detection():
    """
    A class for detecting lenses in an image using a trained model.

    Attributes:
        model: The trained model.
        model_config: The configuration of the model.
        tile: The tile identifier.
        path: The path to the image directory.
        image: The image data.
        bands: The bands to use for detection.
    """
    def __init__(self, model, tile, path = None, bands = ('VIS', 'H', 'J', 'Y')):
        """
        Initializes the Detection class with the provided model, tile, path, and bands.

        Args:
            model: The path to the model.
            tile: The tile identifier.
            path: The path to the image directory. Defaults to None.
            bands: The bands to use for detection. Defaults to ('VIS', 'H', 'J', 'Y').

        Returns:
            None
        """
        self.model = load_model(model)
        self.tile = tile
        self.path = pathlib.Path(path)
        self.image = FitsData(tile, path, bands)
        self.bands = bands

    def detect(self,
               mag_limit = 22,
               save_fits = True,
               draw_boxes = True,
               threshold = 0.97):
        """
        Detect lenses in the image using the provided model.

        Args:
            mag_limit (int, optional): The magnitude limit for lens detection. Defaults to 22.
            export (bool, optional): Whether to export the detected lenses. Defaults to True.
            save_fits (bool, optional): Whether to save the cutout images. Defaults to True.
            draw_boxes (bool, optional): Whether to draw boxes around the detected lenses. 
            Defaults to True.
            threshold (float, optional): The confidence threshold for detection. Defaults to 0.97.

        Returns:
            dict: A dictionary containing the detected lenses. 
            Each lens is represented by a key-value pair, 
            where the key is the object ID and the value is a dictionary
            containing the following information:
                - 'ra' (str): The right ascension of the lens.
                - 'dec' (str): The declination of the lens.
                - 'mag' (str): The magnitude of the lens.
                - 'cert' (float): The confidence score of the detection.
        """

        print(len(self.image.objects))
        lenses = {}
        input_shape = self.model.input_shape
        i = 0
        for object_id in track(self.image.objects.keys(),
                               description=f"Detecting lenses for tile {self.tile}"):
            if (self.image.objects[object_id]['kron_rad'] < min(input_shape[1:3])*0.7
                and 15 < self.image.objects[object_id]['mag'] < mag_limit):
                i+=1
                cutout = self.image.get_tf_tensor(object_id, input_shape[1:3])
                result = self.model.predict(cutout, verbose=0)
                if result[0][0] > threshold:
                    lenses[str(object_id)] = {'ra': str(self.image.objects[object_id]['ra']),
                                            'dec': str(self.image.objects[object_id]['dec']),
                                            'mag': str(self.image.objects[object_id]['mag']),
                                            'cert': str(result[0][0])}

                    if save_fits:
                        self.image.save_fits(cutout[0, :, :, 0],
                                             'lenses_' + str(self.tile) + '/cutout_'+
                                             str(object_id)+'.fits')
        self.export_lenses(lenses)
        if draw_boxes:
            coords = []
            for _, infos in lenses.items():
                coords.append((infos['ra'], infos['dec']))
            self.image.draw_boxes(coords, box_size=200,
                             thickness=2,
                             value=np.max(cutout))
        return lenses

    def export_lenses(self, lenses):
        """
        Exports the detected lenses to a YAML file.

        Args:
            lenses (dict): A dictionary containing the detected lenses.
            Each lens is represented by a key-value pair, where the key is the object ID
            and the value is a dictionary containing the following information:
                - 'ra' (str): The right ascension of the lens.
                - 'dec' (str): The declination of the lens.
                - 'mag' (str): The magnitude of the lens.
                - 'cert' (float): The confidence score of the detection.
        """
        with open('lenses_'+self.tile+'.yml', 'w', encoding='utf-8') as outfile:
            yaml.dump(lenses, outfile, sort_keys=False)
