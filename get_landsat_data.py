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
import precompiled_func


# SHARED_DATA_FOLDER = Path('sample_data')
SHARED_DATA_FOLDER = Path('/project2/macs30123/project_landsat/landsat_scenes')
all_scenes = [folder for folder in listdir(SHARED_DATA_FOLDER) if 'LC08' in folder]

com_areas = util.read_spatial("data/com_areas_chi", "32616")
com_areas = com_areas[["area_numbe", "community", "geometry"]]
com_areas["number"] = pd.to_numeric(com_areas["area_numbe"])
com_areas = com_areas.sort_values(by=["number"])

band_names = ['SR_B1','SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', \
              'SR_B6', 'SR_B7', 'ST_B10']


def create_features():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        scenes = util.subset_scenes_by_year(all_scenes)
    else:
        scenes = None
    scenes = comm.scatter(scenes, root=0)

    band_path = []
    early_sum_scenes = []
    early_sum_LST = []
    late_sum_scenes = []
    late_sum_LST = []
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

        (ndvi, ndsi, ndbi, albedo, awei, gemi, LST) = precompiled_func.caclulate_features_from_landsat(b1, b2, b3, b4, b5, b6, b7, b10)

        # ndvi = np.where((b4+b5)==0, 0, (b5-b4)/(b5+b4))
        # ndsi =np.where((b3+b6)==0, 0, (b3-b6)/(b3+b6))
        # ndbi = np.where((b6+b5)==0, 0, (b6-b5)/(b6+b5))
        # albedo = ((0.356*b1)+(0.1310*b2)+(0.373*b3)+\
        #                 (0.085*b4)+(0.072*b5)-0.0018)/1.016
        # awei = 4*(b3-b6)-(0.25*b5 + 2.75*b6)
        # eta = (2*(b5**2-b4**2) + 1.5*b5 + 0.5*b4) / (b5+b4+0.5)
        # gemi = eta*(1-0.25*eta) - ((b4-0.125)/(1-b4))   
        pattern = '^(?:[^_]+_){3}([^_ ]+)'
        date = re.findall(pattern, SCENE)[0]
        month = date[4:6]
        # prop_veg = np.where((max(ndvi)- min(ndvi))==0, 0, (ndvi - min(ndvi)) / (max(ndvi)- min(ndvi))**2)
        # LSE = (0.004 * prop_veg) + 0.986
        # LST = (b10 / (1 + (10.895 * (b10/14380)) * (np.log(LSE))))

        columns = (ndvi, ndsi, ndbi, albedo, awei, gemi)

        scene_features = np.concatenate(columns, axis=1)
        if month in ['05', '06', '07']: #early summer 
            early_sum_scenes.append(scene_features)
            early_sum_LST.append(LST)   
        else: #late summer
            late_sum_scenes.append(scene_features)
            late_sum_LST.append(LST)
    es_features = util.aggregate_arrays_in_a_period(early_sum_scenes)
    ls_features = util.aggregate_arrays_in_a_period(late_sum_scenes)
    es_LST = util.compute_max(early_sum_LST)
    ls_LST = util.compute_max(late_sum_LST)
    es_features = np.append(es_features, es_LST, axis=1)
    ls_features = np.append(ls_features, ls_LST, axis=1)
    W = util.get_W(com_areas)
    es_features_lag = util.compute_spatial_lag(es_features, W)
    ls_features_lag = util.compute_spatial_lag(ls_features, W)

    early_summer = np.ones(ndvi.shape)
    late_summer = np.full(shape=ndvi.shape, fill_value=2)
    community = com_areas["number"].astype(int).values.reshape(ndvi.shape)
    es_features_lag = np.append(es_features_lag, community, axis=1) 
    ls_features_lag = np.append(ls_features_lag, community, axis=1)
    es_features_lag = np.append(es_features_lag, early_summer, axis=1)
    ls_features_lag = np.append(ls_features_lag, late_summer, axis=1)

    features = np.append(es_features_lag, ls_features_lag, axis=0) 
    year = np.full(shape=(features.shape[0],1),fill_value=int(date[0:4]))
    features = np.append(features, year, axis=1)
  
    n, p = features.shape
    features_all = None
    if rank == 0:
        features_all = np.empty([n*size, p], dtype='float')
    comm.Gather(sendbuf=features, recvbuf=features_all, root=0)
    if rank == 0:
        df = pd.DataFrame(features_all, columns = ['ndvi', 'ndsi', 'ndbi', 'albedo', 'awei', 'gemi', 'LST', 'ndvi_lag', 'ndsi_lag','ndbi_lag', 'albedo_lag', 'awei_lag','gemi_lag', 'LST_lag','community', 'period', 'year'], dtype=object )
        df.to_csv('df.csv')




def main():
    create_features()


if __name__ == '__main__':
    main()



