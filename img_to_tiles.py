# convert geoTiff images to tiles
from pathlib import Path

# import matplotlib.pyplot as plt
from PIL import Image
import rasterio
from tqdm import tqdm

fp = Path('/Users/ben/Downloads/swiss-map-raster25_2015_1328_krel_1.25_2056.tif')
TILE_WIDTH = 128
TILE_HEIGHT = 128

with rasterio.open(fp) as img:
    # rasterio.plot.show(img)
    img_np = img.read()
    img_w = img.width
    img_h = img.height

    output_dir = Path('./tiles/') / fp.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # split into tiles
    for y in tqdm(range(0, img_w, TILE_HEIGHT)):
        for x in range(0, img_h, TILE_WIDTH):
            m = Image.fromarray(img_np[:, x:x+TILE_HEIGHT, y:y+TILE_HEIGHT].transpose(1,2,0))
            m.save(output_dir / f'tile_{x}_{y}.png')


    # save each tile


