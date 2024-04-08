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
# code used to pull the physcial address for given lattitude and longitude 

def getCityState(lat, lon): 
    
    url=f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}2&sensor=false&key=AIzaSyBT0P3Fhfcm99BxNl83Lu4QQWjxSM9pKdM"
    
    #url = f"https://rest.isric.org/soilgrids/v2.0/classification/query?lon={lon}&lat={lat}&number_classes=0"
    response = requests.get(url)
    data= response.json()
   
    json_str = json.dumps(data, indent=4, sort_keys=True)
    if len(data['results'][0]['address_components']) >=3:
        county = (data['results'][0]['address_components'][3]['long_name'] )
    else:
        county=''
        
    address =  data['results'][0]['formatted_address'] 
    return ([county, address])
 
 


names= ['Type','Name','PFAS','Latitude','Longtitude']
datafile =pd.read_csv("https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/CleanTestData.csv",  header=0, index_col=False, names=names)
address=[]
county=[]

for index, row in datafile.iterrows():
    SiteLatt= row['Latitude']
    SiteLong= row['Longtitude']
    #print(SiteLatt, SiteLong)
 
    loc=(getCityState(SiteLatt, SiteLong))
 
    county.append(loc[0])   
    address.append(loc[1])
    
 

datafile['address']=address
datafile['county']=county
datafile.to_csv("withaddress.csv")


