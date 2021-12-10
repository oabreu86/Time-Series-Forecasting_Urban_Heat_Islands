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
from mpi4py import MPI
import util


SHARED_DATA_FOLDER = Path('sample_data')
all_scenes = [folder for folder in listdir(SHARED_DATA_FOLDER) if 'LC08' in folder]

#Need these on all cores
com_areas = util.read_spatial("data/com_areas_chi", "32616")
com_areas = com_areas[["area_numbe", "community", "geometry"]]

band_names = ['SR_B1','SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', \
              'SR_B6', 'SR_B7', 'ST_B10']


def create_features():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # scenes= None 
    # recv_scenes = None
    if rank == 0:
        # scenes = np.array_split(np.array(all_scenes), size)
        # scenes_by_yr = {}
        # for scene in all_scenes:
        #     year = scene[17:21]
        #     scenes_by_yr[year] = scenes_by_yr.get(year, [])+[scene]
        # scenes  = scenes_by_yr.values()
        scenes = util.subset_scenes_by_year(all_scenes)
    else:
        scenes = None
    
    band_path = []
    scenes = comm.scatter(scenes, root=0)
    # print("rank", rank, "scenes_size", scenes.shape)
    # print("rank", rank, "after_scenes", scenes)
    features = None
    for SCENE in scenes:
        scene_path = SHARED_DATA_FOLDER/SCENE
        for band in band_names:
            path = scene_path/"{}_{}.TIF".format(SCENE, band)
            band_path.append(path)
        b1 = util.compute_zonal_stats(band_path[0], com_areas, "b1")
        b2 = util.compute_zonal_stats(band_path[1], com_areas, "b2")
        b3 = util.compute_zonal_stats(band_path[2], com_areas, "b3")
        b4 = util.compute_zonal_stats(band_path[3], com_areas, "b4")
        b5 = util.compute_zonal_stats(band_path[4], com_areas, "b5")
        b6 = util.compute_zonal_stats(band_path[5], com_areas, "b6")
        b7 = util.compute_zonal_stats(band_path[6], com_areas, "b7")
        b10 = util.compute_zonal_stats(band_path[7], com_areas, "b10")

        ndvi = np.where((b4+b5)==0, 0, (b5-b4)/(b5+b4))
        ndsi =np.where((b3+b6)==0, 0, (b3-b6)/(b3+b6))
        ndbi = np.where((b6+b5)==0, 0, (b6-b5)/(b6+b5))
        albedo = ((0.356*b1)+(0.1310*b2)+(0.373*b3)+\
                        (0.085*b4)+(0.072*b5)-0.0018)/1.016
        awei = 4*(b3-b6)-(0.25*b5 + 2.75*b6)
        eta = (2*(b5**2-b4**2) + 1.5*b5 + 0.5*b4) / (b5+b4+0.5)
        gemi = eta*(1-0.25*eta) - ((b4-0.125)/(1-b4))   
        community = com_areas["area_numbe"].astype(int).values.reshape(ndvi.shape)
        pattern = '^(?:[^_]+_){3}([^_ ]+)'
        date = re.findall(pattern, SCENE)[0]
        period = np.full(shape=ndvi.shape,fill_value=date)

        prop_veg = (ndvi - min(ndvi)) / (max(ndvi)- min(ndvi))**2
        LSE = (0.004 * prop_veg) + 0.986
        LST = (b10 / (1 + (10.895 * (b10/14380)) * (np.log(LSE))))

        #not adding period to columns yet since it would break mpi code
        columns = (ndvi, ndsi, ndbi, albedo, awei, gemi, community)

        scene_features = np.concatenate(columns, axis=1)
        #run spatial lag and aggregation code on scene_features

        if features is not None:
            features = np.append(features, scene_features, axis=0)
        else:
            features = scene_features
    print("rank", rank,  "features_size", features.shape)


    n, p = features.shape
    features_all = None
    if rank == 0:
        features_all = np.empty([n*size, p], dtype='float')
    comm.Gather(sendbuf=features, recvbuf=features_all, root=0)
    if rank == 0:
        print("features_all_shape", features_all.shape)
        print("features_all", features_all)


def main():
    create_features()


if __name__ == '__main__':
    main()



