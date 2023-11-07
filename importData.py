#import libraries 

import pandas as pd

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from math import sin, cos, sqrt, atan2, radians
from scipy import spatial
import requests

...
def getElevation(lat, lon):
    url = f"https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}"
    r = requests.get(url)
    data = r.json()
    #print(data)
    return data['results'][0]['elevation']


# function find distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # approximate radius of Earth in kilometers

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    
    return distance

# load landfill
def loaddataset(fileurl, columnnames):
# Load dataset
    dataset = read_csv(fileurl, 
                            names=columnnames, 
                            on_bad_lines='skip',
                            header=None,
                            index_col=False,
                            skiprows=1)

    dataset = dataset.reset_index()
    dataset['Latitude'] = pd.to_numeric(dataset['Latitude'], errors='coerce')
    dataset['Longtitude'] = pd.to_numeric(dataset['Longtitude'], errors='coerce')
    return dataset

#calculate distance between two array of latitude and longtitude
def get_distance(dataset1, dataset2):
    distance = []
    distancetopoint=[]
    for index, row in dataset1.iterrows():
        SiteLatt=row['Latitude']
        SiteLong=row['Longtitude']
        distance = []
        for index, row in dataset2.iterrows():
            distance.append(haversine(row['Latitude'],row['Longtitude'],  SiteLatt,SiteLong))
        distancetopoint.append(min(distance))
    return distancetopoint

# calculate distance 
def calculatedistance(dataset, SiteLatt, SiteLong):
    distance = []
    for index, row in dataset.iterrows():
        distance.append(haversine(row['Latitude'],row['Longtitude'],  SiteLatt,SiteLong))
    return min(distance)

def calculatedistanceandmore(dataset, SiteLatt, SiteLong, fieldtopull):
    distance = []
    for index, row in dataset.iterrows():
        distance.append(haversine(row['Latitude'],row['Longtitude'],  SiteLatt,SiteLong) )
    
    mindistance=min(distance)
    distancefield = dataset.iloc[distance.index(mindistance)][fieldtopull]
    return([mindistance,distancefield])

#calculate closest point between two data sets 


# calculate distance 
# Load airport data set and find a distance 
airportfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Airports.csv"
airportnames = ['city', 'FAA', 'IATA', 'ICAO', 'Airport_Name', 'Role', 'Enplanements','Latitude','Longtitude']
airportdataset= loaddataset(airportfileurl, airportnames)

# Load Landfill data set and find a distance 
landfillfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Permitted_Solid_Waste_Landfills.csv"
landfillcolumnnames = ['FID','Permit_ID','Permit_Nam','Address','City','State','Zip','County','PrimaryWas','PrimaryOpe','PermitStat','PermitExpD','Status','Latitude','Longtitude']
landfilldataset= loaddataset(landfillfileurl, landfillcolumnnames)

# Chem Factory list 
chemfactoryfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Chem%20FactoryList.csv"
chemfactorycolumnnames = ['YEAR','TRIFD','FRS ID','FACILITY NAME','Address','City','County','State','Zip','BTA','TRIBE','Latitude','Longtitude']
chemfactorydataset= loaddataset(chemfactoryfileurl, chemfactorycolumnnames)

# NC firefighting foam usage 
firefightingfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/NCFirefighterClassBFoamUsage.csv"
firefightingcolumnnames = ['Fire Department','Chief','Address','City-State-Zip', 'County','Phone','FDID','Latitude','Longtitude']
firefightingdataset= loaddataset(chemfactoryfileurl, chemfactorycolumnnames)

# NC Water ways 
waterwaysfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/8_Digit_HUC_Subbasins_and_10_Digit_HUC_Watersheds.csv"
waterwayscolumnnames = ['OBJECTID1','HUC_10','HU_10_NAME','DWQ_Basin','Area_sqmi','TOTAL_POP','POP_2010','POP_2000','POP_CHG_10_20','POP_CHG_00_10','ACRES','Latitude','Longtitude']
waterwaysdataset= loaddataset(waterwaysfileurl, waterwayscolumnnames)


# EPA Spill Data
spilldatafileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/SpillData.csv"
spilldatacolumnnames = ['Responsible Company','Latitude','Longtitude', 'State','Water Reached?', 'count']
spilldatadataset= loaddataset(spilldatafileurl, spilldatacolumnnames)

# EPA Chem Industry list
chemindusfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/EPAChemIndustry.csv"
cheminduscolumnnames = ['Facility','Industry', 'State','Latitude','Longtitude',  'count']
chemindusdataset= loaddataset(chemindusfileurl, cheminduscolumnnames)

# EPA Federal site tested positive
fedsitesfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/EPAFederalSitePostiveTest.csv"
fedsitescolumnnames = ['Site Name', 'State','Latitude','Longtitude',  'count']
fedsitesdataset= loaddataset(fedsitesfileurl, fedsitescolumnnames)

# EPA Green house emission
grnhousefileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/EPAGreenHouseGas.csv"
grnhousecolumnnames = ['Site Name', 'State','Latitude','Longtitude',  'count']
grnhousedataset= loaddataset(grnhousefileurl, grnhousecolumnnames)

# NC Solid Waster
solidwastefileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Permitted_Solid_Waste_Septage_Facilities_(SLAS_or_SDTF).csv"
solidwastecolumnnames = ['X','Y','ObjectId','SLAS_Number','NCS_Number','Address','City','County','Date_Orig_Permitted','Date_Permit_Issued','Date_Permit_Expires','Acres','Gallons','Capacity','Domestic_Septage','Grease','Portable_Toilet','Other_Waste','Status','Latitude','Longtitude']
solidwastedataset= loaddataset(solidwastefileurl, solidwastecolumnnames)

# NC DEQ UST incidents https://data-ncdenr.opendata.arcgis.com/datasets/ncdenr::ust-incidents/explore?showTable=true
ustincidentfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/UST_Incidents.csv"
ustincidentcolumnnames = ['Incident name','Latitude','Longtitude', 'total']
ustincidentdataset= loaddataset(ustincidentfileurl, ustincidentcolumnnames)

#Coal Ash
coalashfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Coal_Ash_Structural_Fills__CCB___CLOSED_.csv"
coalashcolumnnames = ['X','Y','FID','Location_I','Site_Name','Address1','Address2','City','State','Zip','County','Start_Date','Latitude','Longtitude','GlobalID']
coalashdataset= loaddataset(coalashfileurl, coalashcolumnnames)

#Hazard Waste
hazardwastefileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Hazardous_Waste_Sites.csv"
hazardwastecolumnnames = ['X','Y','FID','HANDLER_ID','SITE_NAME','LOC_STR_NO','LOC_ADDR_1','LOC_ADDR_2','LOC_CITY','LOC_COUNTY','LOC_ZIP','CONTACT_NA','CONTACT_PH','GENERATOR','TRANSPORTE','TREATER','STORER','LAND_UNIT','HSWA_PERMI','Latitude','Longtitude','HCS_CODE','HCS_REF','HCS_RES']
hazardwastedataset= loaddataset(hazardwastefileurl, hazardwastecolumnnames)

#Hazard Waste
hazardsitefileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Inactive_Hazardous_Sites.csv"
hazardsitecolumnnames = ['X','Y','OBJECTID','EPAID','SITENAME','SITEADDR','SITECITY','SITECOUNTY','Latitude','Longtitude','GEOLOC_COD','SOURCE','Land_Use_R','Vol_Cleanu','Laserfiche','Update_Dat']
hazardsitedataset= loaddataset(hazardsitefileurl, hazardsitecolumnnames)




#datafileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Training%20Data.csv"
datafileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/EPAPositiveTestData.csv"
datacolumnnames = ['Site Type','Name','Total PFAS','Latitude','Longtitude']
traingdataset= loaddataset(datafileurl, datacolumnnames)


chemfactorydataset["distancetowaterways"] = get_distance(chemfactorydataset, waterwaysdataset)
landfilldataset["distancetowaterways"] = get_distance(landfilldataset, waterwaysdataset)

#print( chemfactorydataset.head(20))

distancetoirport=[]
distancetolandfil = []
distancetochemfactory=[]
distancetofirefighting=[]
distancetowaterways=[]
spilldata=[]
chemindustry=[]
federalsites=[]
greenhouse=[]
solidwaste=[]
ustincident=[]
chemtowater =[]
coalash=[]
hazardwaste=[]
oldhazardsite=[]
landfiltowater=[]
elevation=[]


# calculate distance and append to the data set 
for index, row in traingdataset.iterrows():
    SiteLatt= row['Latitude']
    SiteLong= row['Longtitude']
    
    #chemindustry.append(calculatedistance(chemindusdataset, SiteLatt,SiteLong ))
    chemdistance=calculatedistanceandmore(chemfactorydataset, SiteLatt,SiteLong,"distancetowaterways" )
    chemindustry.append(chemdistance[0])
    chemtowater.append(chemdistance[1])

    landfilldistance=calculatedistanceandmore(landfilldataset, SiteLatt,SiteLong,"distancetowaterways" )
    #distancetolandfil.append(calculatedistance(landfilldataset, SiteLatt,SiteLong ))
    distancetolandfil.append(landfilldistance[0])
    landfiltowater.append(landfilldistance[1])

    distancetoirport.append( calculatedistance(airportdataset, SiteLatt,SiteLong ))

    distancetochemfactory.append(calculatedistance(chemfactorydataset, SiteLatt,SiteLong ))
    distancetofirefighting.append(calculatedistance(firefightingdataset, SiteLatt,SiteLong ))
    distancetowaterways.append(calculatedistance(waterwaysdataset, SiteLatt,SiteLong ))
    spilldata.append(calculatedistance(spilldatadataset, SiteLatt,SiteLong ))
    federalsites.append(calculatedistance(fedsitesdataset, SiteLatt,SiteLong ))
    greenhouse.append(calculatedistance(grnhousedataset, SiteLatt,SiteLong ))
    solidwaste.append(calculatedistance(solidwastedataset, SiteLatt,SiteLong ))
    ustincident.append(calculatedistance(ustincidentdataset, SiteLatt,SiteLong ))
    
    coalash.append(calculatedistance(coalashdataset, SiteLatt,SiteLong ))
    hazardwaste.append(calculatedistance(hazardwastedataset, SiteLatt,SiteLong ))
    oldhazardsite.append(calculatedistance(hazardsitedataset, SiteLatt,SiteLong ))
    elevation.append(getElevation(SiteLatt,SiteLong))


# add column to the training set 
traingdataset["distancetoirport"]=distancetoirport
traingdataset["distancetolandfil"]=distancetolandfil
traingdataset["distancetochemfactory"]=distancetochemfactory
traingdataset["distancetofirefighting"]=distancetofirefighting
traingdataset["distancetowaterways"]=distancetowaterways

traingdataset["spilldata"]=spilldata
traingdataset["chemindustry"]=chemindustry
traingdataset["federalsites"]=federalsites
traingdataset["greenhouse"]=greenhouse
traingdataset["solidwaste"]=solidwaste
traingdataset["ustincident"]=ustincident
traingdataset["chemtowater"]=chemtowater
traingdataset["landfiltowater"]=landfiltowater


traingdataset["coalash"]=coalash
traingdataset["hazardwaste"]=hazardwaste
traingdataset["oldhazardsite"]=oldhazardsite

traingdataset["elevation"]=elevation

#save the file 
traingdataset.to_csv('TrainingDataWithDistancenew2.csv', sep=',', encoding='utf-8', index=False)







