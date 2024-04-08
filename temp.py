import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Monthly
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

def corr(x, y, **kwargs):
    
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)
    
def reg_coef(x,y,label=None,hue=None,color=None,**kwargs):
    ax = plt.gca()
    r,p = pearsonr(x,y)
    ax.annotate('r = {:.2f}'.format(r), xy=(0.5,0.5), xycoords='axes fraction', ha='right')

def get_elevation(lat, lon):
    url = f"https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}"
    r = requests.get(url)
    data = r.json()
    print(data)
    elevation = data['results'][0]['elevation']
    print(elevation)
    
def get_soil_type(lat, lon):
    query = ('https://api.opentopodata.org/v1/soil?locations={lat},{lon}').format(lat=lat, lon=lon)
    response = requests.get(query)
    data = response.json()
    print(data)
    soil_type = data['results'][0]['soil_type']
    return soil_type


def get_soil_data(lat, lon):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}"
    response = requests.get(url)
    data= response.json()
    json_str = json.dumps(data, indent=4, sort_keys=True)
    print(json_str)

def get_soil_classification(lat, lon):
    
    url=f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=clay&property=sand&depth=0-5cm&value=mean&value=uncertainty"
    
    #url = f"https://rest.isric.org/soilgrids/v2.0/classification/query?lon={lon}&lat={lat}&number_classes=0"
    response = requests.get(url)
    data= response.json()
    json_str = json.dumps(data, indent=4, sort_keys=True)
    print(json_str)
    clay = data['properties']['layers'][0]['depths'][0]['values']['mean']
    sand = data['properties']['layers'][1]['depths'][0]['values']['mean']
    print( [clay, sand])


def getTemp(lat, lon):
    # Set time period
    start = datetime(2020, 1, 1)
    end = datetime(2022, 12, 31)

    # Create Point for New York, NY
    tpr = Point(lat, lon)

    # Get monthly data for 2022
    data = Monthly(tpr, start, end)
    data = data.fetch()
    
    #WSPD=Wind Speed
    #prcp=Precipitation thenth of mm
    #pres = Pressure
    # return temp, Precipitation, windSpeed, Pressure
    return(data['tavg'].mean(skipna = True), data['prcp'].mean(skipna = True), data['wspd'].mean(skipna = True),data['pres'].mean(skipna = True)   )
    
# Example usage
lat = 27.64
lon = -81.55
#print( get_soil_classification(lat, lon) )
#print(get_elevation(lat, lon))

featurelist=[ 'trial','distancetoirport','distancetolandfil','distancetochemfactory','distancetofirefighting','distancetowaterways',
             'spilldata','chemindustry','greenhouse','solidwaste','ustincident','chemIndustowater','landfiltowater','chemFacttowater',
             'coalash','hazardwaste','oldhazardsite','elevation','countofChemIndus','countofLandfill','countChemFactory','solidwastetowater',
             'countsolidwaste','industryname','ustincidenttowater','counttoustincident','firefightingtowater','counttofirefighting',
             'distancetoprereglandfil','prereglandfiltowater','counttofprereglandfil','fedsitetowater','counttoffedsites','soilclassification',
             'clay','sand','temp','Precipitation','windSpeed','Pressure','distancetowaterdischarge','dischargetowater','countofwaterdischarge',
             'landfillperp','prelandfillperp','WWTPflowrate','distancetofirestation','firestationtowater','countfirestation',
             'landfillcountclosetowater','distancetolandfillwater','cheminduscountclosetowater','distancetochemfacwater',
             'chemfaccountclosetowater','distancetosolidwastewater','solidwastecountclosetowater','distancetofirefightingwater',
             'firefightingcountclosetowater','distancetowaterdischargewater','waterdischargecountclosetowater','septagecountclosetowater',
             'distancetofirestationwater','firestationcountclosetowater','distancetoustincidentwater','ustincidentcountclosetowater',
             'distancetofedsitewater','fedsitecountclosetowater']
feature=pd.read_csv("C:\\Users\\glrao\\pfasmodel\\pfasmodel\\feature.csv", names=featurelist, header=0, index_col=False)

feature.drop(feature[feature.trial == 0].index, inplace=True)

sns.set(rc={'figure.figsize':(6.7,5.27)})

coeffile=pd.DataFrame()
coeffile['feature']=featurelist
cooeflist=[] 
for i in featurelist:
 
    g = sns.lineplot(x='trial', y=i,data=feature,lw=5)
    coef = np.corrcoef(feature['trial'], feature[i])[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    cooeflist.append(label)
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)
    
    plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
    plt.rcParams['axes.titlepad'] = -64  # pad is in points...
    plt.title('Parameter = '+i, fontdict=font)
    plt.xlabel('Parts per Trillion (PPT)', fontdict=font)
    plt.ylabel('Parameter Importance', fontdict=font)
    plt.show()

coeffile['coef']=cooeflist

coeffile.to_csv("coeffile.csv")
    
# visualize the relationship between the features and the response using scatterplots


#https://dev.meteostat.net/python/#installation

