# Nino Index Phase Analysis
The [Nino Index Phase Analysis](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015WR017644) supports the detection of climate teleconnections by categorizing years depending on the climate phase and by selecting, for each phase, the preseason SST anomalies statistically significantly correlated with local conditions. The latter are aggregated via Principal Component Analysis, and the detected teleconnection is validated by using a linear model predicting the local conditions (precipitation) as a function of the 1st PC extracted from the selected SSTs.

### How to install
1. Download and install Anaconda
2. Create a python virtual environment (we tested Python 3.6)
3. Install the following packages using conda-forge (`conda install -n python36 -c conda-forge PACKAGE`)
	- pydap
	- pandas
	- matplotlib
	- scipy
4. Install the following packages from Anaconda-Navigator
	- basemap
	- basemap-data-hires 



