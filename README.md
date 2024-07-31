# GravNet

This package allows users to handle euclid data and detect objects with a CNN in these data. This package was developped in the case of an internship at CNRS/Euclid consortium.

## Installation

In the downloaded directory use ``pip install -e .``

## Basic usages

### Files structure

To use this package you must respect the following file structure.
```
├── U
│   └── EUC_MER_BGSUB-MOSAIC-DECAM-U_TILE.fits
├── Z
│   └── EUC_MER_BGSUB-MOSAIC-DECAM-Z_TILE.fits
├── I
│   └── EUC_MER_BGSUB-MOSAIC-DECAM-I_TILE.fits
├── R
│   └── EUC_MER_BGSUB-MOSAIC-DECAM-R_TILE.fits
├── G
│   └── EUC_MER_BGSUB-MOSAIC-DECAM-G_TILE.fits
├── H
│   └── EUC_MER_BGSUB-MOSAIC-NIR-H_TILE.fits
├── J
│   └── EUC_MER_BGSUB-MOSAIC-NIR-J_TILE.fits
├── Y
│   └── EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE.fits
├── VIS
│   └── EUC_MER_BGSUB-MOSAIC-VIS_TILE.fits
└── CAT
    └── EUC_MER_FINAL-CAT_TILE.fits
```


### detecting lenses in Euclid data with a CNN

```
from gravnet.detection.detection import Detection

detector = Detection('model_resnet50_multi.h5', '102012409', 'euclid_images', bands=['VIS'])

res = detector.detect(save_fits=False, mag_limit=22)
```

### colorize VIS images with DESI (or other ground based telescopes) data

```
from gravnet import ColorData

color_vis = ColorData(tile = 101542818, path = 'euclid_images', )

color_vis.compute_color(coords = (150.2659698, 2.1178021), size = (200, 200), stretch = 'log', save_file = True)
```


## Documentation

The documentation is available in code as doctsring.

## License

Gravnet is licensed under a  CC BY-NC 4.0 license - see the LICENSE in the repo.

## TO DO

- Docsphinx
- Readthedoc
- Packaging on Pypi
- Augment channels combination for coloring

Feel free to contact me for any suggestions or comments.