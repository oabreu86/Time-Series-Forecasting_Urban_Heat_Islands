# Getting Warmer?: Forecasting Chicago Urban Heat Islands 2013-2020

##### Syeda Jaisha, Onel Abreu, and Gabe Morrison
##### Professor Jon Clindaniel
##### MACS 30123: Large Scale Computing for the Social Sciences
##### Friday, December 10, 2021

## Repo Organization

## Link to Video

We made a short video (<5 minutes) summarizing this project! You can see it here!

## Introduction

Temperatures across the world are rising. Large urban centers are particularly susceptible to disproportionate temperature rises. The “Urban Heat Island Effect” (UHI) describes the phenomenon of higher temperatures in urban zones compared with surrounding suburban and rural areas. UHIs tend to occur in areas with little vegetation, near freeways, and high levels of impervious surfaces (Chun and Guldmann, 2014).The consequences of extreme heat events worsened by UHI can include illness and death. 

UHI and extreme heat waves are particularly relevant in the Chicago context; in the summer of 1995, an estimated 739 Chicagoans, mostly from low-income neighborhoods, died as a consequence of a heat wave (Abreu, 2021). UHIs remain a pressing concern for Chicago. The Environmental Protection Agency reports that in high emissions scenarios, Chicago could experience an additional 30 days per year that reach temperatures above 100℉ (EPA). 
Cities induce UHIs through various channels. Cities tend to have highly concentrated consumption of energy resources as a consequence of their clustering of many people. Some of this energy dissipates as heat. Large urban structures can also trap some of this heat. On a local scale, buildings, green spaces, and pavement impact local temperature (Gago et al. 2013). 

In the long term, cities can take a number of responses based on expanding green spaces and making alternative urban design interventions. Cities can encourage urban greening by planting urban trees, creating and expanding parks, and encouraging green and roofs. Some research also indicates cities can use “cold” pavement materials. Cities can also use zoning to change their urban form. Research suggests that this could improve shade and mitigate solar radiation (Gago et al. 2013). 

However, these are long-term responses. UHIs, particularly during warm summers like 1995, reflect imminent threats., and cities like Chicago could use advanced knowledge of communities likely to be particularly hard-hit by UHIs. This project seeks to respond to that need. 

More specifically, we use LandSat 8 data imagery to identify LST and also other facets of the built and natural environment at the Chicago Community Area scale. We then build a set of Machine Learning models to predict a neighborhood’s LST in future years given previous years’ data. Our study period was 2013-2020. We conducted analysis at the Chicago Community Area level. This type of modelling could be useful for the City of Chicago in that model predictions from a given year could help identify the extent to which UHIs will impact certain neighborhoods and take short-term policy actions in response. The spatial scale is also salient given that Chicago leaders and residents both have good conceptions of their neighborhood as determined by Community Area. 

The rest of this write-up is as follows. In the Large Scale Data solution, we describe our data more explicitly. We also outline the pipeline we used to convert raw Landsat data to usable features in an ML context. In the Machine Learning section, we detail our ML models. In our Results section, we describe the results of our Machine Learning, and we offer a brief Conclusion section. 



## Large Scale Data Solution

## Results

## Conclusion

## Contributions

We all contributed to the writing of the report. Onel focused on handling the Machine Learning components. Gabe and Jaisha both worked on the pre-processing pipeline. Gabe focused on handling the spatial data (creating weights matrix and spatial lag columns, converting raster to vector data). Jaisha focused on the parallelization and speed-up of the code (MPI4Py, AOT Numba compilation). 


## Work Cited 

Abreu, Onel. “Predictive Heat and Vulnerable Populations.” Informational Memorandum for Mayor Lori E. Lightfoot. August, 2021. https://docs.google.com/document/d/1uvt__e_tAJAMejR_Gf1IVLI8SO3K4g_f/edit

Avdan, U. and Jovanovska, G., 2016. Algorithm for Automated Mapping of Land Surface Temperature Using LANDSAT 8 Satellite Data. Journal of Sensors, 2016, pp.1-8.

Chun, B. and J.-M. Guldmann. “Spatial Statistical analysis and Simulation of the Urban Heat Island in High-Density Central Cities.” Landscape and Urban Planning, Volume 125, 2014, pp. 76-88. https://www.sciencedirect.com/science/article/pii/S0169204614000243?via%3Dihub

Cochrane, Courtney. “Time Series Nested Cross-Validation.” Medium, Towards Data Science, May 2018, https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9. 

Feyisa, Gudina L.; Meilby, Henrik; Fensholt, Rasmus; Proud, Simon R. (2014). Automated Water Extraction Index: A new technique for surface water mapping using Landsat imagery. Gago, E.J., J. Roldan, R. Pacheco-Torres, J. Ordóñez. “The city and urban heat islands: A review of strategies to mitigate adverse effects.” Renewable and Sustainable Energy Reviews, Volume 25, 2013, pp. 749 - 758. https://www.sciencedirect.com/science/article/pii/S1364032113003602?casa_token=KX7obYCeU4YAAAAA:9Orkvp9USAi67kO8GTz1G-4VsoOMcihcO7MvIPwM-nTMQnQaSSN7o55qQXji1K8BawI3TEK9KA#f

Liang, S. 2000. “Narrowband to broadband conversions of land surface albedo I algorithms.” Remote Sensing of Environment 76, 213-238.
Reducing the Urban Heat Island. Green Infrastructure. The United States Environmental Protection Agency. https://www.epa.gov/green-infrastructure/reduce-urban-heat-island-effect

Remote Sensing of Environment, 140(), 23–35. doi:10.1016/j.rse.2013.08.029 

Pro.arcgis.com. 2021. GEMI—ArcGIS Pro | Documentation. [online] Available at: <https://pro.arcgis.com/en/pro-app/latest/arcpy/image-analyst/gemi.htm > [Accessed 11 December 2021].

Pro.arcgis.com. 2021. NDBI—ArcGIS Pro | Documentation. [online] Available at: <https://pro.arcgis.com/en/pro-app/latest/arcpy/image-analyst/ndbi.htm  > [Accessed 11 December 2021].

Sayler, Kristi. “Landsat 8 Collection 2 (C2) Level 2 Science Project (L2SP) Guide.” U.S. Geological Survey, The Department of the Interior. Version 2.0, September 2020. https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1619_Landsat8-C2-L2-ScienceProductGuide-v2.pdf

Usgs.gov. 2021. Landsat Normalized Difference Vegetation Index | U.S. Geological Survey. [online] Available at: <https://www.usgs.gov/landsat-missions/landsat-normalized-difference-vegetation-index> [Accessed 15 November 2021].

Usgs.gov. 2021. Landsat Normalized Difference Snow Index | U.S. Geological Survey. [online] Available at:
<https://www.usgs.gov/landsat-missions/normalized-difference-snow-index > [Accessed 15 November 2021]
