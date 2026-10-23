"""Ground height (metres above sea level) anywhere in the study area.

Source: Copernicus DEM GLO-30, free 30 m elevation data from the European
Space Agency, hosted on AWS Open Data (no account needed). One tile covers
1 x 1 degree; N10 E077 covers all of Munnar.

We keep a small crop of the tile as a plain text grid so the project (and the
Streamlit demo) doesn't need rasterio at runtime. Make the crop once with
`python -m hillsense.elevation --crop` (that step needs `pip install rasterio`).

Heights between grid points are filled in with bilinear interpolation, so a
road gets a smooth profile instead of 30 m "steps".
"""

import sys

import numpy as np

TILE_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N10_00_E077_00_DEM/"
    "Copernicus_DSM_COG_10_N10_00_E077_00_DEM.tif"
)
TILE_PATH = "data/dem/cop30_N10_E077.tif"
CROP_PATH = "data/dem/munnar_dem.txt"
CROP_BBOX = (77.00, 10.00, 77.22, 10.12)  # west, south, east, north


class Elevation:
    def __init__(self, grid, west, north, xres, yres):
        self.grid = grid.astype(float)
        self.west, self.north = west, north
        self.xres, self.yres = xres, yres  # degrees per pixel (yres is negative)

    @classmethod
    def from_txt(cls, path=CROP_PATH):
        with open(path) as f:
            west, north, xres, yres = map(float, f.readline().lstrip("# ").split())
        return cls(np.loadtxt(path), west, north, xres, yres)

    def at(self, lat, lon):
        """Bilinear-interpolated height at lat/lon (scalars or arrays). NaN outside the grid."""
        lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        # pixel values describe the CENTRE of each pixel, hence the -0.5
        col = (lon - self.west) / self.xres - 0.5
        row = (lat - self.north) / self.yres - 0.5
        r0, c0 = np.floor(row).astype(int), np.floor(col).astype(int)
        fr, fc = row - r0, col - c0
        h, w = self.grid.shape
        inside = (r0 >= 0) & (c0 >= 0) & (r0 + 1 < h) & (c0 + 1 < w)
        r0c, c0c = np.clip(r0, 0, h - 2), np.clip(c0, 0, w - 2)
        g = self.grid
        z = (
            g[r0c, c0c] * (1 - fr) * (1 - fc)
            + g[r0c, c0c + 1] * (1 - fr) * fc
            + g[r0c + 1, c0c] * fr * (1 - fc)
            + g[r0c + 1, c0c + 1] * fr * fc
        )
        return np.where(inside, z, np.nan)


def crop_tile(tile_path=TILE_PATH, out_path=CROP_PATH, bbox=CROP_BBOX):
    """Cut the study area out of the full 1x1 degree tile (needs rasterio)."""
    import rasterio

    with rasterio.open(tile_path) as src:
        window = src.window(*bbox)
        grid = src.read(1, window=window)
        t = src.window_transform(window)
    np.savetxt(out_path, grid, fmt="%.1f", header=f"{t.c} {t.f} {t.a} {t.e}")
    return grid.shape


if __name__ == "__main__":
    if "--crop" in sys.argv:
        print("cropped grid:", crop_tile())
    dem = Elevation.from_txt()
    print(f"grid {dem.grid.shape}, {dem.grid.min():.0f}-{dem.grid.max():.0f} m")
