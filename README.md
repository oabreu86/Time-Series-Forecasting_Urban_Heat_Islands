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


### Data

We used Tier 1 data from Landsat 8 Collection 2 Level 2. Landsat 8 data is ideal. The satellite has been running since 2013, giving us more than 7 years of consistent data. We use Collection 2 data because the preprocessing systems used to create Collection 2 data are more up-to-date and better documented than those for the Collection 1 data (Sayler 2020).  For a broader comparison between the two collections, see Sayler (2020). Within Collection 2, we used Level 2 data because it was more suited for our data needs. Level 2 gave us Surface Reflectance and Surface Temperature values for each pixel for each band which saved us from manual calculation of these values. Moreover, the pixel values (known as DNs) in Level 2 are already corrected for atmospheric effects. Finally, we only use Tier 1 images because they constitute the highest quality Landsat 8 data available. 

We downloaded Landsat8 data from the USGS Earth Explorer. We initially considered accessing the Landsat data in the AWS Open Data Registry S3 Bucket landsat-pds. However, our AWS Educate accounts’ location on us-east-1 did not permit us to access that bucket as it was located in another AWS region. Given that our target variable was maximum LST which almost certainly happens during the summer and our study period was 2013 - 2020, we only downloaded data from summer months (May - September, inclusive) of those years. We also only downloaded Landsat 8 scenes that covered Chicago; we did this using the USGS Earth Explorer API and by uploading a shapefile containing rough Chicago boundaries. This ultimately left us with 133 scenes across 8 years. Each Landsat8 scene contains at least 8 .TIF (Tagged Image Format) files. Each file represents data from a different “band”. We used the bands as described in the Features Subsection to generate features to predict LST and to create LST itself. 


### Large Scale Data Processing
1. Collecting Data on Midway:
    - We first scp-ed the data we downloaded from our machines to the Midway 2 Computing Cluster. 
    - We stored the data in a directory where each scene had its own subdirectory. Each subdirectory contained all the bands for its respective scene.
    - We used the same names as USGS for the subdirectory; this is noteworthy because the scene date is included in this name.
2. Parallelizing Data Reading and Feature Engineering: 
    - We generated a list of all the scenes for all the years in our data and then split them into different lists by the scene acquisition year. 
    - Using an .sbatch script, we requested 8 cores to run our processes in parallel. We used one core for each year of data we processed. 
    - We used the MPI for Python package to scatter the list of lists of scenes across all cores so that each core would receive a list for all the scenes for a single year and would be responsible for processing the same. 

3. Other Data Pre-processing
    - We wrote code to read a Chicago Community Areas shapefile and convert it to a geopandas geodataframe with CRS 32616 (the same CRS as the TIF data). We then ran this code on each core.
    - Using the libpysal package, we created a spatial weights matrix using a first-order queen's contiguity definition of neighborhood. Since the pysal package stores that data as a dictionary, we converted it into a true matrix (2D numpy array). We also read this file to each core. 
    - Because both files were small and geopandas does not integrate well with MPI4Py, we used this setup instead of reading each file once and then broadcasting to each core. We recognize that this could also be a successful code implementation.		

4. Handling Data on Each Core:
    - Individual Scene Processing:
         - We first read each of the 8 .TIF files that correspond to the bands we need for our features, using the rasterio package.
         - We then preprocessed the band data so that if it was outside the acceptable spectrum range, we converted it to zero. We then multiplied all values by the multiplicative scale factor and added the additive scale factor. For more details and the actual coefficients, see Sayler (2020). 
         - We used the zonal_stats function from the rasterstats package to compute the mean value of each band for all pixels in a community area. 
         - We then calculated the features using the relevant bands. 
     - Collective Scene Processing:
         - We took the cellwise average for each of the feature columns separately for all scenes in May, June and July and for all scenes in August, and September. This converts a set of more than a dozen scenes per year to two averages: one for “early summer” and one for “late summer.”
         - We then computed all of the features in the Features subsection described below (using the formulas we also show) for both the early summer and the late summer scenes. 
         - We created our target Land Surface Temperature (LST) variable by taking the maximum of the early summer and late summer LST for each Community Area. 
         - We calculated the spatial lag of the features for the early summer and late summer matrix. This gives the average value of each feature in the Community Areas surrounding a given Community Area. 
         - As a last pre-processing step, we added data for scene year and period (i.e. whether it was “early summer” or “late summer”) and Community Area number. 

To summarize, this subprocess moves data from many scenes for a given year, each of which contain 8 .TIF files(bands), to two 2D arrays. Each array contains 77 rows, one for each Community Area. Each array also contains 14 columns, one column for each of the features (inclusing LST that is our target variable) shown in the feature section below and one column for the spatial lag of those features.

5. Aggregating the Feature Set:
    - Our intelligent parallelization and aggregation by community area design ensures that each core returns an array with the same shape since the number of Landsat scenes may vary from year to year. This made possible the gathering of all the arrays on a single core after the preprocessing and feature engineering. 
    - We performed a Gather operation to stack together all of the 2D numpy arrays from each of the cores.
    - We converted this to a pandas dataframe and subsequently wrote out the results as a CSV.

### Machine Learning:
Note: We considered using Dask to perform our ML. However, our data is relatively small. After the pre-processing steps, we reduced our data to have 1232 rows (77 Community Areas * 8 years * 2 periods) and 17 columns (7 true features + 7 spatial lags of features+ year+period+community area number). Additionally, the dask_ml libraries that would save us the most computational time through parallelization (GridSearch, Pipeline, CrossValidator) do not fit our time series modeling due to the need for more custom code for those functions. We decided that the communication costs associated with Dask and lack of flexibility for our problem domain with dask_ml were not worth the small parallelization benefit and ran sklearn algorithms locally.

#### Features: 
As alluded to above, we first determine band averages (in multiple senses). First we average all band values for given 30m x 30m cells in the Landsat data to the Community Area level for a given scene. Then we average those Community Area averages into two groupings per year based on month (May and June for early summer and July, August, and September for late summer). 
We use the period and community area averages to compute the estimates of ndvi, ndsi, albedo, awei, eta, and gemi and then determine the spatial lag for each of those variables.
Thus, our feature matrix is organized so that each row refers to an individual Community Area in a given year. Each column is a feature, either one of those described in the Features section or the spatial lag of one of those features in a given period. Thus, a given row has four features associated with, for example, NDVI: NDVI Early Summer, NDVI Late Summer, Spatial Lag of NDVI Early Summer, Spatial Lag of NDVI Late Summer. 

#### Models:
We use a family of linear models. Specifically, we test OLS Regression, Ridge Regression, LASSO Regression, and Elastic Net Regularization. For the last three models, we test a variety of hyperparameters.

#### Training, Cross-Validation, Testing
Given the importance of the time dimension for our ability to predict temperature, our modeling relies on many assumptions included in many traditional time series models, with a few caveats. 
Given the infrequency (one scene every two weeks or so) of the satellite imagery collected across the entire period, we can’t reliably ensure that we have equal amounts of data for different sub periods within our timeframe for training, an important aspect of time series models.
Additionally, traditional k-fold cross validation’s randomized approach to splitting data does not work within our time series problem domain. This is because data leakage is likely to occur due to the temporal dependencies between temperature across time (e.g. the temperature yesterday has a strong impact on the temperature today). Additionally, because the choice of test set is arbitrary in traditional cross-validation, our test set error is likely to be a poor estimate of error on a new, independent set (Cochrane, 2018).
For a given model, we train the data on a given year i and choose optimal hyperparameters by cross-validating the model performance against the predicted LST in year i + 1. 
Finally, with the optimal hyperparameters, we assess model performance in year i + 2. 


### Features
To predict LST, we created features from the raw bands. We computed the following features using band data as shown below:

#### Normalized Difference Vegetation Index (NDVI) 
NDVI = b5 - b4b5 + b4

(Landsat Normalized Difference Vegetation Index | U.S. Geological Survey, 2021)


#### Normalized Difference Snow Index (NDSI)
NDSI = b3 - b6b3 + b6 

(Landsat Normalized Difference Snow Index | U.S. Geological Survey, 2021)


#### Normalized Difference Built-up Index (NDBI)
NDBI = b6 - b5b6 + b5 

(NDBI—ArcGIS Pro | Documentation, 2021)


#### Albedo
albedo = (.356 * b1) + (.1310 * b2) + (.373 * b3) + (.085 * b4) +(0.072*b5)-0.00181.016   

(Liang, 2000)


#### Automated Water Extraction Index (AWEInsh) 
AWEI= 4(b3 - b6)-(.25b5 + 2.75 b6) 

(Gudian et. al,  2014)


#### Global Environmental Monitoring Index (GEMI)
GEMI = eta (1-.25eta) - b4-.1251-b4 where   eta =2*(b52-b42) + 1.5b5 + 0.5b4(b5+b4+0.5) (GEMI—ArcGIS Pro | Documentation, 2021)

#### Land Surface Temperature (LST)
LST=  b101+(10.895(b10/14380))ln(LSE) where LSE = 0.04α +0.986  where α =  (NDVI - min(NDVI)max(NDVI) - min(NDVI)) 2 

(Avdan and Jovanovska, 2016)



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
