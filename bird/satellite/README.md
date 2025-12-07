Sentinel-2 over spectral bands (RGB, NIR, SWIR).

```bash
python3 -m bird.satellite.multispectral_viewer \
  --aoi bay_area \
  --tile-size 512 \
  --bands rgb nir swir16 swir22
```

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

<picture> 
  <img alt="spectral" src="/docs/multispectral.jpg" width="450px""> 
</picture>
