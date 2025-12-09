## Sat

Satellite data is low frame rate, high information density. This folder has
utils to download frames from Sentinel2, which is part of the Copernicus
European space program. Sentinel2 has 10m / pixel resolution on RGB and
different res for the other 10 spectral bands. Data goes back to 2015. One RGB
earth-pass is 22K tiles, 55MB/tile, ~1TB. So a single band is 333GB. Sentinel-2
has 13 bands, rough aprox 4.3TB/earth-pass on all bands.

```bash
# To download, set an aoi in aoi.py and run for a timerange
python3 -m bird.satellite.cli download --aoi microsoft_fairwater --start-date 2024-01-01 --end-date 2024-12-31
```

```bash
python3 -m bird.satellite.multispectral_viewer \
  --aoi bay_area \
  --tile-size 512 \
  --bands rgb nir swir16 swir22
```

<picture> 
  <img alt="spectral" src="/docs/multispectral.jpg"> 
</picture>

| Band | Name (Meaning)                                                                                 | Wavelength (nm) | Resolution | Use                       |
| ---- | ---------------------------------------------------------------------------------------------- | --------------- | ---------- | ------------------------- |
| B01  | Coastal aerosol (detects atmospheric aerosols over water)                                      | 443             | 60m        | Aerosol detection         |
| B02  | Blue (visible blue light)                                                                      | 490             | 10m        | RGB                       |
| B03  | Green (visible green light)                                                                    | 560             | 10m        | RGB                       |
| B04  | Red (visible red light)                                                                        | 665             | 10m        | RGB                       |
| B05  | Red Edge 1 (first red edge band, transition zone between red and NIR, sensitive to vegetation) | 705             | 20m        | Vegetation classification |
| B06  | Red Edge 2 (second red edge band, more into NIR, vegetation monitoring)                        | 740             | 20m        | Vegetation classification |
| B07  | Red Edge 3 (third red edge band, further into NIR, vegetation monitoring)                      | 783             | 20m        | Vegetation classification |
| B08  | NIR (Near Infrared, beyond visible red, strong reflectance from healthy vegetation)            | 842             | 10m        | Vegetation, water         |
| B8A  | NIR Narrow (narrower NIR band for subtle vegetation differences)                               | 865             | 20m        | Vegetation, water         |
| B09  | Water Vapor (detects atmospheric water vapor)                                                  | 945             | 60m        | Atmospheric correction    |
| B10  | Cirrus (detects high-altitude cirrus clouds)                                                   | 1375            | 60m        | Cloud detection           |
| B11  | SWIR1 (Shortwave Infrared 1, sensitive to moisture content)                                    | 1610            | 20m        | Moisture, snow/ice        |
| B12  | SWIR2 (Shortwave Infrared 2, deeper SWIR, detects fires and minerals)                          | 2190            | 20m        | Fire, minerals            |
