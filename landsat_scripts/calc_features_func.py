import numpy as np
from numba.pycc import CC


#name of compiled module to create:
cc = CC('precompiled_func')

@cc.export('caclulate_features_from_landsat', '(f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:])')
def caclulate_features_from_landsat(b1, b2, b3, b4, b5, b6, b7, b10):
    ndvi = np.where((b4+b5)==0, 0, (b5-b4)/(b5+b4))
    ndsi =np.where((b3+b6)==0, 0, (b3-b6)/(b3+b6))
    ndbi = np.where((b6+b5)==0, 0, (b6-b5)/(b6+b5))
    albedo = ((0.356*b2)+(0.1310*b4)+(0.373*b5)+\
                    (0.085*b6)+(0.072*b7)-0.0018)/1.016
    awei = 4*(b3-b6)-(0.25*b5) + (2.75*b6)
    eta = (2*(b5**2-b4**2) + 1.5*b5 + 0.5*b4) / (b5+b4+0.5)
    gemi = eta*(1-0.25*eta) - ((b4-0.125)/(1-b4))   
    min_ndvi = np.min(ndvi)
    max_ndvi = np.max(ndvi)
    min_ndvi_arr = np.full(shape=ndvi.shape, fill_value=float(min_ndvi))
    max_ndvi_arr = np.full(shape=ndvi.shape, fill_value=float(max_ndvi))
    if max_ndvi - min_ndvi == 0:
        prop_veg = np.full(shape=ndvi.shape, fill_value=0.0)
    else:
        prop_veg =  ((ndvi - 0.2) / (0.3))**2
        # prop_veg =  ((ndvi - min_ndvi_arr) / (max_ndvi_arr-min_ndvi_arr))**2
    LSE = (0.004 * prop_veg) + 0.986
    LST = (b10 / (1 + (10.895 * (b10/14380)) * (np.log(LSE))))

    return (ndvi, ndsi, ndbi, albedo, awei, gemi, LST)

cc.compile()
