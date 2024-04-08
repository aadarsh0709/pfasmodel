import matplotlib.pyplot as plt
from soilgrids import SoilGrids
import pandas as pd
import pyproj 

 
# +proj=utm +zone=17 +datum=WGS84 +units=ft +no_defs 32617
# +proj=longlat +datum=WGS84 +no_defs
#https://mygeodata.cloud/cs2cs/


def lonlat_to_xy(lon, lat):
    proj_latlon = pyproj.Proj(proj='latlong',datum='WGS84')
    proj_xy = pyproj.Proj(proj="utm", zone=17, datum='WGS84')
    xy = pyproj.transform(proj_latlon, proj_xy, lon, lat)
    return xy[0], xy[1]

  
def xy_to_lonlat(x, y):
    proj_latlon = pyproj.Proj(proj='latlong',datum='WGS84')
    proj_xy = pyproj.Proj(proj="utm", unit='ft', zone=17, datum='WGS84')
    lonlat = pyproj.transform(proj_xy, proj_latlon, x, y)
   
    return lonlat[0], lonlat[1]

print(lonlat_to_xy(36.006601,-79.904711))