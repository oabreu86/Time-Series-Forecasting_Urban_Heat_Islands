import os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
from shapely.geometry import Point
from pathlib import Path
import re
from rasterio.mask import mask
import json
import datetime
from os import listdir
from rasterstats import zonal_stats



def read_spatial(path, espg_code):
    '''
    Function to read spatial data and converts to ESPG 3435
    Input: path to the data file
            epgs_code (string): epsg code as a string
    Output: a gpd object 
    '''
    epsg = "EPSG:" + espg_code
    file=gpd.read_file(path)
    file=file.to_crs(epsg)
    # print(path, file.crs)
    return file

def subset_scenes_by_year(all_scenes):
    '''
    Input: list of all Landsat scenes
    Output: list of subsets of scenes divided by acquisition year
    '''
    scenes_by_yr = {}
    for scene in all_scenes:
        year = scene[17:21]
        scenes_by_yr[year] = scenes_by_yr.get(year, [])+[scene]
    
    return scenes_by_yr.values()

def compute_zonal_stats(path_to_raster, vector, band_name):
    '''
    Inputs: 
        path_to_raster (string): path to raster data
        vector (geopandas df): 
        band_name (string): name of band
    Outputs:
        nparray (1D np array): mean values of the raster data 
                             ordered by com area order
    '''
    col_name = "mean_" + band_name
    sum_stats = zonal_stats(vector, path_to_raster, 
                            # nodata = Nan,
                            stats=["mean"])
    df = pd.DataFrame(sum_stats)
    df = df.rename(columns = {"mean": col_name})
    nparray = np.array(df)
    
    return nparray
