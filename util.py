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
import libpysal as lp


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

def aggregate_arrays_in_a_period(list_of_arrays):
    '''
    Function that takes the mean of all arrays in list_of_arrays
    
    Inputs:
        list_of_arrays (list of arrays (in this case 2D))
    Outputs:
        mean_array (2D np.array): the cell average of the arrays.
        Note 1 : this treats missing values as 0
        Note 2: Despite its name, it really is a nice array :) 
    '''
    n, p = list_of_arrays[0].shape
    a_agg = np.zeros((n, p))
    
    for a in list_of_arrays:
        a = np.nan_to_num(a)
        a_agg += a
    
    a_mean = a_agg / len(list_of_arrays)
    
    return a_mean

def compute_max(list_of_arrays):
    '''
    Function that computes the elementwise max of the arrays in the list of arrays
    Inputs:
        list_of_arrays (list of 1D np arrays), note the arrays should be the same size
    Output:
        max_array (1D np array): the elementwise max from each array
    '''
    n, p = list_of_arrays[0].shape
    mx = np.zeros((n, p))  
    for i in range(len(list_of_arrays)):
        mx = np.maximum(list_of_arrays[i], mx)
    
    return mx

def get_W(com_areas):
    '''
    Function that takes community areas shapefile (should be sorted by area number)
    Input:
        com_areas (gpd): a community areas shapefile sorted by area number
    Output:
        W (2D np.array): the spatial weights matrix
    
    '''
    #Get pysal weights matrix:
    weights_matrix = lp.weights.Queen.from_dataframe(com_areas, idVariable='number')

    #Access their dictionary
    w_dict = weights_matrix.neighbors
    W = np.zeros((77, 77))

    for com_area, list_of_neighbors in w_dict.items():
        for neighbor in list_of_neighbors:
            W[int(com_area) -1, int(neighbor) -1] = 1

    #Standardize:
    sum_of_rows = W.sum(axis=1)
    W = W / sum_of_rows[:, np.newaxis]
    
    return W

def compute_spatial_lag(data_array, W):
    '''
    Computes the first order queen contiguity spatial lag for data in the data_array given weights matrix W
    Inputs:
        data_array (2D np.array): data matrix with rows indicating community areas (ordered by com area number)
        W (2D np.array): a spatial weights matrix of size 77 x 77
    Outputs:
        data_array (2D np.array): the data array, but with columns appended for each spatial lag
    '''
    n, p = data_array.shape
    new_data_array = data_array
    
    for i in range(p):
        col = data_array[:,i]
        lag = W @ col
        lag = lag.reshape(-1, 1)
        new_data_array = np.hstack((new_data_array, lag))
    return new_data_array

