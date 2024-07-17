# GravNet

This package allow users to handle euclid data and detect objects with a CNN in these data.

## Installation

In the downloaded directory use ``pip install -e .``

## Basic usages

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