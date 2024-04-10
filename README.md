# Solar Forecasting Using Weather and Location Data

This project explores a photovoltaic (PV) dataset with a view to investigating its potential for forecasting power output. The dataset is taken from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/northern-hemisphere-horizontal-photovoltaic) and accompanies the paper [*Machine Learning Modeling of Horizontal Photovoltaics Using Weather and Location Data*](https://www.mdpi.com/1996-1073/13/10/2570) by Pasion et al. 

## Motivation and aims
Solar energy is a key renewable resource and its adoption is forecast to grow steeply as the price of PV falls. The numerous benefits of solar adoption include the reduction of air pollution by replacing traditional fuel sources like wood and coal, thereby combating the climate crisis while improving health, stimulus to economies by creating employment opportunities and contributing to economic growth, and providing access to modern and reliable energy services in developing countries. 

Much of this growth is anticipated to be in distributed PV, i.e. horizontal cells installed on home- and building-owners' roofs, and this is forecasted to grow from 180 TWh to 6–10 TWh by 2050. As such, it is important to be able to reliably predict the output of such cells in order to integrate them with supply grids and support the transition to greener economies.

Initial aims of the project are as follows.
1. EDA, with interactive plots that could potentially serve as the basis for a dashboard.
2. Replication of the results of the paper using standard machine learning algorithms.
3. Investigation of how far ahead of time it is possible to reliably forecast the power output with the given data.
4. As a possible extension, augmentation of the data with forecasts with an aim to reliably predict 24 hours in advance, which is about what would be required for the power to be a viable energy source for the grid. Success on this point will depend largely on the quantity and quality of the data for inference purposes and the results of the above steps.

## Dataset
It contains 14 months of data taken at 12 Northern Hemisphere locations having varying climatic conditions; in total, we have 21045 data points. Data recorded was related to the following: 
- Location 
- Date 
- Time sampled 
- Altitude
- Humidity
- Ambient temperature
- Wind speed
- Visibility
- Pressure
- Cloud ceiling
- Power output.

An interesting aspect of the dataset is that it was taken in the absence of irradiation measurements, which is known to be a valuable predictor of power output: since irradiation can be time-consuming to measure and difficult to forecast with a high degree of certainty, the idea was to investigate the potential for accurate forecasting using more readily available and reliable measurements. 

Below is a screen grab showing the locations of the different PV sites across North America. The pop-ups on each location show the average power output and can be set to an hourly or monthly average.

![Map](./images/map_example.png)

Examination of the dataset revealed the following summary points. See the EDA notebook for details and to explore the data further. 

### Range of data collection and missing values
At each location, the data ranges from 10:00 to 15:45 at 15-minute intervals from 23 May 2017 to 4 Oct 2018 (499 days in total). Although there are no explicit nulls in the dataset, the data is patchy depending on the particular location, i.e. we don't have consistent data available for every 15 minutes throughout every day for any of the sites. The sparsity of data is visualised below using comparison to a hypothetical ideal situation of available readings for every 15-minute interval from 10:00 to 15:45 for a given site. We can see that even in the best-case scenario we have only 23% of the data, which would likely limit the use of the dataset for time-series forecasting.

![Missing values](./images/missing_values.png)

### General variation of power and other features
The power largely exhibits approximately bimodal distributions with the locations of the central peaks varying from site to site and with peak frequency occurring at high power in some cases (e.g. USAFA) and low power in others (e.g. Malmstrom). 

![Power by location](./images/power_by_loc.png)

There is also interesting variation in average monthly power output depending on the location: at some sites there is a smooth increase towards the middle of the year while at others there is much more variation between adjacent months. Some kind of cyclical encoding will be necessary here to ensure that, e.g., the values recorded in December are 'close' to those in January for modelling purposes.

Temperature, pressure and wind speed show broad distributions with varying degrees of skew. Transforms to make the distributions more normal may help at the modelling stage. In contrast, the cloud ceiling and visibility both have a high peak to large values indicating that encoding the data as, e.g., high, medium and low may be advantageous when modelling. 

### Correlation matrix
The following is a correlation matrix of the features, which shows a mixture of positive and negative correlations with power. Note that (unsurprisingly) pressure and altitude are perfectly correlated with one another and don't vary with location. Consequently, the latter was discarded on the basis that the former showed more variation with power output.

<img src="./images/correlation_matrix.png" width="700" height="600" />

These correlations can also be confirmed visually using a scatterplot as with, for example, the broadly positive correlation between temperature and power at USAFA shown below.

<img src="./images/power_vs_temp.png" width="700" height="600" /> 

## Summary 
The above EDA serves as an initial step ahead of predictive modelling using the dataset. The data is clean and validated by comparison with the source paper. The next step is to obtain a baseline model using the current features followed by iterative feature engineering and selection, and comparison with the results of the paper.

#### References
1. C. Pasion et al. *Energies* **2020**, 13, 2570; [doi:10.3390/en13102570](https://www.mdpi.com/1996-1073/13/10/2570)
2. Project Drawdown [*Distributed Solar Voltaics*](https://drawdown.org/solutions/distributed-solar-photovoltaics#:~:text=Distributed%20solar%20photovoltaics%20(PV)%20are,and%20natural%20gas%20power%20plants.)