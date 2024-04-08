#import libraries 

import pandas as pd
import numpy as np

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
from geopy.geocoders import Nominatim
from sklearn.preprocessing import LabelEncoder
from meteostat import Point, Monthly
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# variable to count many KM radius the contamination resources
distancetocount = 15

IndustryName=''
# change this to adjust PPT 
PFASLevelCutOff = 4

# flag to include soil data or not 
addsoil = True


#remove nonNC Flag
removenonNC=True

def getTemp(lat, lon):
    # Set time period
    start = datetime(2020, 1, 1)
    end = datetime(2022, 12, 31)

    # Create Point  
    tpr = Point(lat, lon)

    # Get monthly data for 2022
    data = Monthly(tpr, start, end)
    data = data.fetch()
    
    #WSPD=Wind Speed
    #prcp=Precipitation thenth of mm
    #pres = Pressure
    # return temp, Precipitation, windSpeed, Pressure
    return(data['tavg'].mean(skipna = True), data['prcp'].mean(skipna = True), data['wspd'].mean(skipna = True),data['pres'].mean(skipna = True)   )

...
def getElevation(lat, lon):
    url = f"https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}"
    r = requests.get(url)
    data = r.json()
    #print(data)
    return data['results'][0]['elevation']

def get_soil_classification(lat, lon):
    url = f"https://rest.isric.org/soilgrids/v2.0/classification/query?lon={lon}&lat={lat}&number_classes=0"
    response = requests.get(url)
    data= response.json()
    return( data['wrb_class_name'] )

def get_claySand(lat, lon):
    url=f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=clay&property=sand&depth=0-5cm&value=mean&value=uncertainty"
    
    #url = f"https://rest.isric.org/soilgrids/v2.0/classification/query?lon={lon}&lat={lat}&number_classes=0"
    response = requests.get(url)
    data= response.json()
    clay = data['properties']['layers'][0]['depths'][0]['values']['mean']
    sand = data['properties']['layers'][1]['depths'][0]['values']['mean']
    return ( [clay, sand])
    

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
                            index_col=False,
                            encoding='unicode_escape',
                            header=0, 
                            skiprows=1)

    dataset = dataset.reset_index()
    dataset['Latitude'] = pd.to_numeric(dataset['Latitude'], errors='coerce')
    dataset['Longtitude'] = pd.to_numeric(dataset['Longtitude'], errors='coerce')
    return dataset

#calculate distance between two array of latitude and longtitude
def get_distancetowaterways(dataset1, dataset2):
    distance = []
    distancetopoint=[]
    position=[]
    for index, row in dataset1.iterrows():
        SiteLatt=row['Latitude']
        SiteLong=row['Longtitude']
        distance = []
        for index, row2 in dataset2.iterrows():
            distance.append(haversine(row2['Latitude'],row2['Longtitude'],  SiteLatt,SiteLong)) 
        distancetopoint.append(min(distance))
        position.append(distance.index(min(distance)))
    return [distancetopoint, position]

def get_distance(dataset1, dataset2):
    distance = []
    distancetopoint=[]
    for index, row in dataset1.iterrows():
        SiteLatt=row['Latitude']
        SiteLong=row['Longtitude']
        distance = []
        for index, row2 in dataset2.iterrows():
            distance.append(haversine(row2['Latitude'],row2['Longtitude'],  SiteLatt,SiteLong)) 
        distancetopoint.append(min(distance))
 
    return distancetopoint 

# calculate distance 
def calculatedistance(dataset, SiteLatt, SiteLong):
    distance = []
    for index, row in dataset.iterrows():
        distance.append(haversine(row['Latitude'],row['Longtitude'],  SiteLatt,SiteLong))
    return min(distance)

def calculatedistanceandmore(dataset, SiteLatt, SiteLong, fieldtopull=None, radiuscount=distancetocount):
    tempIndexPosition =-1
    distance = []
    for index, row in dataset.iterrows():
        distance.append(haversine(row['Latitude'],row['Longtitude'],  SiteLatt,SiteLong) )
    
    y = np.array(distance)
    mindistance=min(distance)
    position=distance.index(mindistance)
    # get the position of the min distance from distance array and pull the distance to water way from same position
    if fieldtopull != None:
        distancefield = dataset.iloc[position][fieldtopull]
    else:
        distancefield=0

    # return min distance , ditance to water, count within radius, position of match 
    # store the index position in global variable so we can pull the industry name 
    return([mindistance,distancefield, np.count_nonzero(y <= radiuscount ), position])


#calculate closest point between two data sets 

print("Loading the key data files.")
# calculate distance 
# Load airport data set and find a distance 
airportfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Airport.csv"
airportnames = ['City served','FAA','IATA','ICAO','Airport name','Enplanements (2019)','Latitude','Longtitude']
airportdataset= loaddataset(airportfileurl, airportnames)
print ( " Total Numer of Airports identified with coordinates in NC =", airportdataset.shape[0])


# Load Landfill data set and find a distance 
landfillfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Permitted_Solid_Waste_Landfills.csv"
landfillcolumnnames = ['FID','Permit_ID','Permit_Nam','Address','City','State','Zip','County','PrimaryWas','PrimaryOpe','PermitStat','PermitExpD','Status','Latitude','Longtitude']
landfilldataset= loaddataset(landfillfileurl, landfillcolumnnames)
print ( " Total Numer of inactive Closed Landfills identified with coordinates in NC =", landfilldataset[landfilldataset.Status =='Closed'].shape[0])
print ( " Total Numer of active Landfills identified with coordinates in NC =", landfilldataset[landfilldataset.Status =='Active'].shape[0])

precipitationlf=[]
print("Getting Landfill Percipitation Data..")

# calculate distance and append to the data set 
for index, row in landfilldataset.iterrows():

    SiteLatt= row['Latitude']
    SiteLong= row['Longtitude']
    # get weather information 
    weather=getTemp(SiteLatt,SiteLong)
    precipitationlf.append(weather[1])
landfilldataset['precipitation']  = precipitationlf

# Chem Factory list 
chemfactoryfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Chem%20FactoryList.csv"
chemfactorycolumnnames = ['YEAR','TRIFD','FRS ID','FACILITY NAME','Address','City','County','State','Zip','BTA','TRIBE','Latitude','Longtitude']
chemfactorydataset= loaddataset(chemfactoryfileurl, chemfactorycolumnnames)
if removenonNC:
    chemfactorydataset.drop(chemfactorydataset[chemfactorydataset.State != 'NC'].index, inplace=True)
print ( " Total Numer of EPA tracked chemical factories coordinates in NC =", chemfactorydataset.shape[0])

# NC firefighting foam usage 
firefightingfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/NCFirefighterClassBFoamUsage.csv"
firefightingcolumnnames = ['Fire Department','Chief','Address','City-State-Zip', 'County','Phone','FDID','Latitude','Longtitude']
firefightingdataset= loaddataset(chemfactoryfileurl, chemfactorycolumnnames)
print ( " Total Numer of sites tracked by NCPFAS Network where firefighting foam used in NC =", firefightingdataset.shape[0])

# NC Water ways 
#waterwaysfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/NC_DEQ%2012_Digit_HUC_Subwatersheds.csv"
#waterwayscolumnnames = ['OBJECTID1','HUC_8','HUC_10','HUC_12','ACRES','HU_10_NAME','HU_12_NAME','Basin','DWQ_Basin','Population','Shape__Area','Shape__Length','POP_2010','POP_2000','AREA','POP_CHG_10_20','POP_CHG_00_10','Latitude','Longtitude']
waterwaysfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/NCWaterWaysTopoQuest.csv"
waterwayscolumnnames = ['Name','Type','Latitude','Longtitude']
waterwaysdataset= loaddataset(waterwaysfileurl, waterwayscolumnnames)


#remove dam lake and reservoir 
waterwaysdataset.drop(waterwaysdataset[waterwaysdataset.Type == 'Dam'].index, inplace=True)
waterwaysdataset.drop(waterwaysdataset[waterwaysdataset.Type == 'Lake'].index, inplace=True)
 
wwcount = waterwaysdataset.groupby('Type').count()
print(wwcount)

# EPA Spill Data
spilldatafileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/SpillData.csv"
spilldatacolumnnames = ['Responsible Company','Latitude','Longtitude', 'State','Water Reached?', 'count']
spilldatadataset= loaddataset(spilldatafileurl, spilldatacolumnnames)
if removenonNC:
    spilldatadataset.drop(spilldatadataset[spilldatadataset.State != 'NC'].index, inplace=True)

print ( "EPA Spill data site count in NC", spilldatadataset.shape[0])


# EPA Chem Industry list
chemindusfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/EPAChemIndustry.csv"
cheminduscolumnnames = ['Facility','Industry', 'State','Latitude','Longtitude',  'count']
chemindusdataset= loaddataset(chemindusfileurl, cheminduscolumnnames)
if removenonNC:
    chemindusdataset.drop(chemindusdataset[chemindusdataset.State != ' NC'].index, inplace=True)
print ( "EPA tracked industry list  in NC", chemindusdataset.shape[0])
chemcount = chemindusdataset.groupby('Industry').count()
print(chemcount)

# EPA Federal site tested positive
fedsitesfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/EPAFederalSitePostiveTest.csv"
fedsitescolumnnames = ['Site Name', 'State','Latitude','Longtitude',  'count']
fedsitesdataset= loaddataset(fedsitesfileurl, fedsitescolumnnames)
if removenonNC:
    fedsitesdataset.drop(fedsitesdataset[fedsitesdataset.State != ' NC'].index, inplace=True)
print ( "EPA tracked federal sites tested positive in NC", fedsitesdataset.shape[0])

# EPA Green house emission
#grnhousefileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/EPAGreenHouseGas.csv"
#grnhousecolumnnames = ['Site Name', 'State','Latitude','Longtitude',  'count']

grnhousefileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/NCDEQGreenHouseGasPermits.csv"
grnhousecolumnnames=['Facility Name','Latitude','Longtitude']
grnhousedataset= loaddataset(grnhousefileurl, grnhousecolumnnames)
print ( "NC DEQ Title V permits issued sites in NC  ", grnhousedataset.shape[0])


# NC Solid Waster
solidwastefileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Permitted_Solid_Waste_Septage_Facilities_(SLAS_or_SDTF).csv"
solidwastecolumnnames = ['X','Y','ObjectId','SLAS_Number','NCS_Number','Address','City','County','Date_Orig_Permitted','Date_Permit_Issued','Date_Permit_Expires','Acres','Gallons','Capacity','Domestic_Septage','Grease','Portable_Toilet','Other_Waste','Status','Latitude','Longtitude']
solidwastedataset= loaddataset(solidwastefileurl, solidwastecolumnnames)

print ( "Permitted_Solid_Waste_Septage_Facilities_(SLAS_or_SDTF) in NC ", solidwastedataset.shape[0])


# NC DEQ UST incidents https://data-ncdenr.opendata.arcgis.com/datasets/ncdenr::ust-incidents/explore?showTable=true
ustincidentfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/UST_Incidents.csv"
ustincidentcolumnnames = ['Incident name','Latitude','Longtitude', 'total']
ustincidentdataset= loaddataset(ustincidentfileurl, ustincidentcolumnnames)
print ( "NC DEQ UST Incidents count  ", ustincidentdataset.shape[0])


#Coal Ash
coalashfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Coal_Ash_Structural_Fills__CCB___CLOSED_.csv"
coalashcolumnnames = ['X','Y','FID','Location_I','Site_Name','Address1','Address2','City','State','Zip','County','Start_Date','Latitude','Longtitude','GlobalID']
coalashdataset= loaddataset(coalashfileurl, coalashcolumnnames)
coalashdataset.drop(coalashdataset.loc[coalashdataset['State'] != 'NC'].index, inplace=True)
print ( "NC DEQ coal ash storage facilities count  ", ustincidentdataset.shape[0])


#Hazard Waste
hazardwastefileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Hazardous_Waste_Sites.csv"
hazardwastecolumnnames = ['X','Y','FID','HANDLER_ID','SITE_NAME','LOC_STR_NO','LOC_ADDR_1','LOC_ADDR_2','LOC_CITY','LOC_COUNTY','LOC_ZIP','CONTACT_NA','CONTACT_PH','GENERATOR','TRANSPORTE','TREATER','STORER','LAND_UNIT','HSWA_PERMI','Latitude','Longtitude','HCS_CODE','HCS_REF','HCS_RES']
hazardwastedataset= loaddataset(hazardwastefileurl, hazardwastecolumnnames)
print ( "NC DEQ active Hazardous Waste Sites count  ", hazardwastedataset.shape[0])

#Inactive Hazard Waste
hazardsitefileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/Inactive_Hazardous_Sites.csv"
hazardsitecolumnnames = ['X','Y','OBJECTID','EPAID','SITENAME','SITEADDR','SITECITY','SITECOUNTY','Latitude','Longtitude','GEOLOC_COD','SOURCE','Land_Use_R','Vol_Cleanu','Laserfiche','Update_Dat']
hazardsitedataset= loaddataset(hazardsitefileurl, hazardsitecolumnnames)
print ( "NC DEQ inactive Hazardous Waste Sites count  ", hazardsitedataset.shape[0])

#NC DEQ pre regulatory landfills 
prereglandfilfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/NC%20DEQ%20Pre-Regulatory_Landfill_Sites.csv"
prereglandfilcolumnnames = ['X','Y','OBJECTID','EPAID','SITENAME','SITEADDR','SITECITY','SITECOUNTY','Latitude','Longtitude','source','GEOLOC_COD','STATUS','PRO_CONTAC','PHONE','EMAIL','NFA_RESTRI','Update_Dat','Doc_Link']
prereglandfildataset= loaddataset(prereglandfilfileurl, prereglandfilcolumnnames)
print ( "NC DEQ pre-regulatory landfill sites count   ", prereglandfildataset.shape[0])

precipitationplf=[]
print("Getting Pre-regulatory Landfill Percipitation Data..")

# calculate distance and append to the data set 
for index, row in prereglandfildataset.iterrows():

    SiteLatt= row['Latitude']
    SiteLong= row['Longtitude']
    # get weather information 
    weather=getTemp(SiteLatt,SiteLong)
    precipitationplf.append(weather[1])
prereglandfildataset['precipitation']  = precipitationplf

#NC DEQ Water discharge permits 
waterdischargefileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/NPDES_Wastewater_Discharge_Permits.csv"
waterdischargecolumnnames = ['X','Y','ObjectId','PERMITNUMBER','PERMIT_TYPE','PERMIT_STATUS','ORIGINAL_ISSUED_DT','PERMIT_EFFECTIVE_DATE','PERMIT_EXPIRATION_DT','FACILITY','FACILITY_ACTIVE','FACILITY_STATUS','OWNER','OWNER_TYPE','MAJOR','COUNTY','REGION','ASBUILTFLOWQTYGPD','Latitude','Longtitude']
waterdischargedataset= loaddataset(waterdischargefileurl, waterdischargecolumnnames)
print ( "NPDES Wastewater Discharge Permits count ( non residential ). ex: WWTP, Chemical industry   ", waterdischargedataset.shape[0])


#NC DEQ Septage sites 
septagefileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/NCDEQSeptage_Sites.csv"
septagecolumnnames = ['X','Y','Date_Orig','Date_Issue','Date_Expir','FID','Permit_ID','NCS_Number','Firm_Name','Contact_FN','Contact_LN','Phone','Address','City','State','Zip','County','Owns','Domestic_S','Grease','Portable_T','Other_Wast','Capacity','Other_Wa_1','Capacity_D','Latitude','Longtitude']
septagedataset= loaddataset(septagefileurl, septagecolumnnames)
print ( "NPDEQ Permitted Septage Sites ", septagedataset.shape[0])

# NC Firestations list 
firestationfileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/NC_Fire_Stations.csv"
firestationecolumnnames = ['X','Y','OBJECTID','FD_ID','DEPT_NAME','STATION_NUMBER','STATION_ADDRESS','CITY','STATE','COUNTY','ZIP_CODE','Latitude','Longtitude']
firestationdataset= loaddataset(firestationfileurl, firestationecolumnnames)
print ( "NC One map lisst of firestations docummented = ", firestationdataset.shape[0])



print("Loading the actual test result file.")

datafileurl = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/CleanTestDataFinal.csv"
datacolumnnames = ['Type','Name','PFAS','Latitude','Longtitude']
traingdataset= loaddataset(datafileurl, datacolumnnames)

 
print('Total numer of PFAS positive site information gathered for this project ', traingdataset.shape[0])

# add a column to flag if PFAS is found or not
traingdataset['PFASFound'] = np.where(traingdataset['PFAS'] > PFASLevelCutOff, 1, 0)

# instead of just counting distance b/w landfill to waterways we will have count landfills close to the waterway point like lake reservoir etc 
# For example if there is multiple chemical factories close to river and the well is situated close to that river probablity of that wayter way 
# being the source for contamination and transport is very hight 

# take each water way position then count how many contamination are close to that point 
# calculate distance and append to the data set 
print('Calculating the water contamination site counts near the water ways')
landfillcountclosetowater=[]
Landfillclosetowaterposition=[]
distancetolandfillwater = []
distancetocheminduswater = []
cheminduscountclosetowater=[]
chemindusclosetowaterposition=[]
distancetochemfacwater = []
chemfaccountclosetowater=[]
chemfacclosetowaterposition=[]
distancetosolidwastewater = []
solidwastecountclosetowater=[]
solidwasteclosetowaterposition=[]
distancetofirefightingwater = []
firefightingcountclosetowater=[]
firefightingclosetowaterposition=[]
distancetowaterdischargewater = []
waterdischargecountclosetowater=[]
waterdischargeclosetowaterposition=[]
distancetoseptagewater = []
septagecountclosetowater=[]
septageclosetowaterposition=[]
distancetofirestationwater = []
firestationcountclosetowater=[]
firestationclosetowaterposition=[]
distancetoustincidentwater = []
ustincidentcountclosetowater=[]
ustincidentclosetowaterposition=[]
distancetofedsitewater = []
fedsitecountclosetowater=[]
fedsiteclosetowaterposition=[]


for index, row in waterwaysdataset.iterrows():
    # iterage through the waterways list populate the count of contamination within given distance
    SiteLatt= row['Latitude']
    SiteLong= row['Longtitude']
   
    # get the landfill information close to this waterway
    result = calculatedistanceandmore(landfilldataset, SiteLatt, SiteLong, None, distancetocount)
    distancetolandfillwater.append(result[0])
    landfillcountclosetowater.append(result[2])
    Landfillclosetowaterposition.append(result[3] )
    
  
    # get the landfill information close to this waterway
    result = calculatedistanceandmore(chemindusdataset, SiteLatt, SiteLong, None, distancetocount)
    distancetocheminduswater.append(result[0])
    cheminduscountclosetowater.append(result[2])
    chemindusclosetowaterposition.append(result[3] )
    
    # get the landfill information close to this waterway
    result = calculatedistanceandmore(chemfactorydataset, SiteLatt, SiteLong, None, distancetocount)
    distancetochemfacwater.append(result[0])
    chemfaccountclosetowater.append(result[2])
    chemfacclosetowaterposition.append(result[3] )
    
    # get the landfill information close to this waterway
    result = calculatedistanceandmore(solidwastedataset, SiteLatt, SiteLong, None, distancetocount)
    distancetosolidwastewater.append(result[0])
    solidwastecountclosetowater.append(result[2])
    solidwasteclosetowaterposition.append(result[3] )
    
    # get the landfill information close to this waterway
    result = calculatedistanceandmore(firefightingdataset, SiteLatt, SiteLong, None, distancetocount)
    distancetofirefightingwater.append(result[0])
    firefightingcountclosetowater.append(result[2])
    firefightingclosetowaterposition.append(result[3] )    
    
    # get the landfill information close to this waterway
    result = calculatedistanceandmore(waterdischargedataset, SiteLatt, SiteLong, None, distancetocount)
    distancetowaterdischargewater.append(result[0])
    waterdischargecountclosetowater.append(result[2])
    waterdischargeclosetowaterposition.append(result[3] )
    
    # get the landfill information close to this waterway
    result = calculatedistanceandmore(septagedataset, SiteLatt, SiteLong, None, distancetocount)
    distancetoseptagewater.append(result[0])
    septagecountclosetowater.append(result[2])
    septageclosetowaterposition.append(result[3] )    
    
    # get the landfill information close to this waterway
    result = calculatedistanceandmore(firestationdataset, SiteLatt, SiteLong, None, distancetocount)
    distancetofirestationwater.append(result[0])
    firestationcountclosetowater.append(result[2])
    firestationclosetowaterposition.append(result[3] )    
    
    # get the landfill information close to this waterway
    result = calculatedistanceandmore(ustincidentdataset, SiteLatt, SiteLong, None, distancetocount)
    distancetoustincidentwater.append(result[0])
    ustincidentcountclosetowater.append(result[2])
    ustincidentclosetowaterposition.append(result[3] )    
    
    # get the landfill information close to this waterway
    result = calculatedistanceandmore(fedsitesdataset, SiteLatt, SiteLong, None, distancetocount)
    distancetofedsitewater.append(result[0])
    fedsitecountclosetowater.append(result[2])
    fedsiteclosetowaterposition.append(result[3] )    
    
waterwaysdataset['landfillcountclosetowater']=landfillcountclosetowater
waterwaysdataset['Landfillclosetowaterposition']=Landfillclosetowaterposition
waterwaysdataset['distancetolandfillwater']=distancetolandfillwater

waterwaysdataset['distancetocheminduswater']=distancetocheminduswater
waterwaysdataset['chemfaccountclosetowater']=chemfaccountclosetowater
waterwaysdataset['chemfacclosetowaterposition']=chemfacclosetowaterposition

waterwaysdataset['distancetochemfacwater']=distancetochemfacwater
waterwaysdataset['cheminduscountclosetowater']=cheminduscountclosetowater
waterwaysdataset['chemindusclosetowaterposition']=chemindusclosetowaterposition

waterwaysdataset['distancetosolidwastewater']=distancetosolidwastewater
waterwaysdataset['solidwastecountclosetowater']=solidwastecountclosetowater
waterwaysdataset['solidwasteclosetowaterposition']=solidwasteclosetowaterposition

waterwaysdataset['distancetofirefightingwater']=distancetofirefightingwater
waterwaysdataset['firefightingcountclosetowater']=firefightingcountclosetowater
waterwaysdataset['firefightingclosetowaterposition']=firefightingclosetowaterposition

waterwaysdataset['distancetowaterdischargewater']=distancetowaterdischargewater
waterwaysdataset['waterdischargecountclosetowater']=waterdischargecountclosetowater
waterwaysdataset['waterdischargeclosetowaterposition']=waterdischargeclosetowaterposition

waterwaysdataset['distancetoseptagewater']=distancetoseptagewater
waterwaysdataset['septagecountclosetowater']=septagecountclosetowater
waterwaysdataset['septageclosetowaterposition']=septageclosetowaterposition

waterwaysdataset['distancetofirestationwater']=distancetofirestationwater
waterwaysdataset['firestationcountclosetowater']=firestationcountclosetowater
waterwaysdataset['firestationclosetowaterposition']=firestationclosetowaterposition

waterwaysdataset['distancetoustincidentwater']=distancetoustincidentwater
waterwaysdataset['ustincidentcountclosetowater']=ustincidentcountclosetowater
waterwaysdataset['ustincidentclosetowaterposition']=ustincidentclosetowaterposition

waterwaysdataset['distancetofedsitewater']=distancetofedsitewater
waterwaysdataset['fedsitecountclosetowater']=fedsitecountclosetowater
waterwaysdataset['fedsiteclosetowaterposition']=fedsiteclosetowaterposition

print("Calculating the distance to waterways to key sites.")
# calculate distance to water ways 
def getDistancetoWaterways(sourcedataset, waterdataset):
    result = get_distancetowaterways(sourcedataset, waterdataset)
    sourcedataset["distancetowaterways"]=result[0]
    sourcedataset["waterwayposition"]=result[1]

#get wateyways distance to contamination site 
getDistancetoWaterways(chemindusdataset, waterwaysdataset)
getDistancetoWaterways(landfilldataset, waterwaysdataset)
getDistancetoWaterways(chemfactorydataset, waterwaysdataset)
getDistancetoWaterways(solidwastedataset, waterwaysdataset)
getDistancetoWaterways(ustincidentdataset, waterwaysdataset)
getDistancetoWaterways(firefightingdataset, waterwaysdataset)
getDistancetoWaterways(fedsitesdataset, waterwaysdataset)
getDistancetoWaterways(waterdischargedataset, waterwaysdataset)
getDistancetoWaterways(septagedataset, waterwaysdataset)
getDistancetoWaterways(firestationdataset, waterwaysdataset)
getDistancetoWaterways(prereglandfildataset, waterwaysdataset)


# now we know the waterway location close the each of contamination sources like landfile
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
chemIndustowater =[]
landfiltowater=[]
chemFacttowater=[]
solidwastetowater=[]
ustincidenttowater=[]
firefightingtowater=[]
coalash=[]
hazardwaste=[]
oldhazardsite=[]

elevation=[]

countofChemIndus=[]
countofLandfill=[]
countChemFactory=[]
countsolidwaste=[]
counttoustincident=[]
counttofirefighting=[]


distancetoprereglandfil=[]
prereglandfiltowater=[]
counttofprereglandfil=[]

industryname=[]

fedsitetowater=[]
counttoffedsites=[]

soilclassification=[]
clay=[]
sand=[]

temp=[]
Precipitation=[]
windSpeed=[]
Pressure=[]

distancetowaterdischarge=[]
dischargetowater=[]
countofwaterdischarge=[]
landfillperp=[]
prelandfillperp=[]
WWTPflowrate = []

distancetofirestation=[]
firestationtowater=[]
countfirestation=[]


# parameter close the waterways near the test site 
landfillcountclosetowater=[]
distancetolandfillwater = []
distancetocheminduswater = []
cheminduscountclosetowater=[]
distancetochemfacwater = []
chemfaccountclosetowater=[]
distancetosolidwastewater = []
solidwastecountclosetowater=[]
distancetofirefightingwater = []
firefightingcountclosetowater=[]
distancetowaterdischargewater = []
waterdischargecountclosetowater=[]
distancetoseptagewater = []
septagecountclosetowater=[]
distancetofirestationwater = []
firestationcountclosetowater=[]
distancetoustincidentwater = []
ustincidentcountclosetowater=[]
distancetofedsitewater = []
fedsitecountclosetowater=[]
    

print("Looping through sites to calculate key parameters.")
count=0
# calculate distance and append to the data set 
for index, row in traingdataset.iterrows():
    
    print('Processing record #', count)
    count=count+1
    
    SiteLatt= row['Latitude']
    SiteLong= row['Longtitude']
    #print(SiteLatt, SiteLong)

    # calculatedistanceandmore return min distance=0 , ditance to water=1, count within radius=2, position of match =3
    # calculate distance to waterways 
    result = calculatedistanceandmore(waterwaysdataset, SiteLatt,SiteLong )

    distancetowaterways.append(result[0])
    waterwayposiotion=result[3]
          
    # now add all waterway parameters to data set 
    landfillcountclosetowater.append(waterwaysdataset.iloc[waterwayposiotion]['landfillcountclosetowater'])
    distancetolandfillwater.append(waterwaysdataset.iloc[waterwayposiotion]['distancetolandfillwater'])
    distancetocheminduswater.append(waterwaysdataset.iloc[waterwayposiotion]['distancetocheminduswater'])
    cheminduscountclosetowater.append(waterwaysdataset.iloc[waterwayposiotion]['cheminduscountclosetowater'])
    distancetochemfacwater.append(waterwaysdataset.iloc[waterwayposiotion]['distancetochemfacwater'])
    chemfaccountclosetowater.append(waterwaysdataset.iloc[waterwayposiotion]['chemfaccountclosetowater'])
    distancetosolidwastewater.append(waterwaysdataset.iloc[waterwayposiotion]['distancetosolidwastewater'])
    solidwastecountclosetowater.append(waterwaysdataset.iloc[waterwayposiotion]['solidwastecountclosetowater'])
    distancetofirefightingwater.append(waterwaysdataset.iloc[waterwayposiotion]['distancetofirefightingwater'])
    firefightingcountclosetowater.append(waterwaysdataset.iloc[waterwayposiotion]['firefightingcountclosetowater'])
    distancetowaterdischargewater.append(waterwaysdataset.iloc[waterwayposiotion]['distancetowaterdischargewater'])
    waterdischargecountclosetowater.append(waterwaysdataset.iloc[waterwayposiotion]['waterdischargecountclosetowater'])
    distancetoseptagewater.append(waterwaysdataset.iloc[waterwayposiotion]['distancetoseptagewater'])
    septagecountclosetowater.append(waterwaysdataset.iloc[waterwayposiotion]['septagecountclosetowater'])
    distancetofirestationwater.append(waterwaysdataset.iloc[waterwayposiotion]['distancetofirestationwater'])
    firestationcountclosetowater.append(waterwaysdataset.iloc[waterwayposiotion]['firestationcountclosetowater'])
    distancetoustincidentwater.append(waterwaysdataset.iloc[waterwayposiotion]['distancetoustincidentwater'])
    ustincidentcountclosetowater.append(waterwaysdataset.iloc[waterwayposiotion]['ustincidentcountclosetowater'])
    distancetofedsitewater.append(waterwaysdataset.iloc[waterwayposiotion]['distancetofedsitewater'])
    fedsitecountclosetowater.append(waterwaysdataset.iloc[waterwayposiotion]['fedsitecountclosetowater'])

    

    #calculate param for water discharge
    waterdischargedistance=calculatedistanceandmore(waterdischargedataset, SiteLatt,SiteLong,"distancetowaterways" )
    distancetowaterdischarge.append(waterdischargedistance[0])
    dischargetowater.append(waterdischargedistance[1])
    countofwaterdischarge.append(waterdischargedistance[2])
    # use iloc to pull the flow 
    WWTPflowrate.append(waterdischargedataset.iloc[waterdischargedistance[3]]['ASBUILTFLOWQTYGPD'] )
           
    #distancetolandfil.append(calculatedistance(landfilldataset, SiteLatt,SiteLong ))
    landfilldistance=calculatedistanceandmore(landfilldataset, SiteLatt,SiteLong,"distancetowaterways" )
    distancetolandfil.append(landfilldistance[0])
    landfiltowater.append(landfilldistance[1])
    countofLandfill.append(landfilldistance[2])
    landfillperp.append( landfilldataset.iloc[landfilldistance[3]]['precipitation'] )

    #chemindustry.append(calculatedistance(chemindusdataset, SiteLatt,SiteLong ))
    chemdistance=calculatedistanceandmore(chemindusdataset, SiteLatt,SiteLong,"distancetowaterways" )
    chemindustry.append(chemdistance[0])
    chemIndustowater.append(chemdistance[1])
    countofChemIndus.append(chemdistance[2])
    #get industry name 
    industryname.append(chemindusdataset.iloc[chemdistance[3]]['Industry'])
 
    
    #distancetochemfactory.append(calculatedistance(chemfactorydataset, SiteLatt,SiteLong ))
    distancetochemfactorydistance=calculatedistanceandmore(chemfactorydataset, SiteLatt,SiteLong,"distancetowaterways" )
    distancetochemfactory.append(distancetochemfactorydistance[0])
    chemFacttowater.append(distancetochemfactorydistance[1])
    countChemFactory.append(distancetochemfactorydistance[2])

    firestationdistance=calculatedistanceandmore(firestationdataset, SiteLatt,SiteLong,"distancetowaterways" )
    distancetofirestation.append(firestationdistance[0])
    firestationtowater.append(firestationdistance[1])
    countfirestation.append(firestationdistance[2])



    #solidwaste.append(calculatedistance(solidwastedataset, SiteLatt,SiteLong ))
    solidwastedistance=calculatedistanceandmore(chemfactorydataset, SiteLatt,SiteLong,"distancetowaterways" )
    solidwaste.append(solidwastedistance[0])
    solidwastetowater.append(solidwastedistance[1])
    countsolidwaste.append(solidwastedistance[2])

    #ustincident.append(calculatedistance(ustincidentdataset, SiteLatt,SiteLong ))
    ustdistance=calculatedistanceandmore(ustincidentdataset, SiteLatt,SiteLong,"distancetowaterways" )
    ustincident.append(ustdistance[0])
    ustincidenttowater.append(ustdistance[1])
    counttoustincident.append(ustdistance[2])
    
    
    #distancetofirefighting.append(calculatedistance(firefightingdataset, SiteLatt,SiteLong ))
    firefightingdistance=calculatedistanceandmore(firefightingdataset, SiteLatt,SiteLong,"distancetowaterways" )
    distancetofirefighting.append(firefightingdistance[0])
    firefightingtowater.append(firefightingdistance[1])
    counttofirefighting.append(firefightingdistance[2])
    
    
    #pre-regulatory landfill 
    fedsitesdistance=calculatedistanceandmore(fedsitesdataset, SiteLatt,SiteLong,"distancetowaterways" )
    federalsites.append(fedsitesdistance[0])
    fedsitetowater.append(fedsitesdistance[1])
    counttoffedsites.append(fedsitesdistance[2])
    
    # federal sitee
    prereglandfildistance=calculatedistanceandmore(prereglandfildataset, SiteLatt,SiteLong,"distancetowaterways" )
    distancetoprereglandfil.append(prereglandfildistance[0])
    prereglandfiltowater.append(prereglandfildistance[1])
    counttofprereglandfil.append(prereglandfildistance[2])
    prelandfillperp.append( landfilldataset.iloc[landfilldistance[3]]['precipitation'] )
    

    # calculate distance 
    distancetoirport.append( calculatedistance(airportdataset, SiteLatt,SiteLong ))   
    spilldata.append(calculatedistance(spilldatadataset, SiteLatt,SiteLong ))
    greenhouse.append(calculatedistance(grnhousedataset, SiteLatt,SiteLong ))
    
    coalash.append(calculatedistance(coalashdataset, SiteLatt,SiteLong ))
    hazardwaste.append(calculatedistance(hazardwastedataset, SiteLatt,SiteLong ))
    oldhazardsite.append(calculatedistance(hazardsitedataset, SiteLatt,SiteLong ))
    elevation.append(getElevation(SiteLatt,SiteLong))
    
    # get weather information 
    weather=getTemp(SiteLatt,SiteLong)
    temp.append(weather[0])
    Precipitation.append(weather[1])
    windSpeed.append(weather[2])
    Pressure.append(weather[3])
    
    if addsoil:
        soilclassification.append(get_soil_classification(SiteLatt,SiteLong))
        # get soil informaiton 
        sc=get_claySand(SiteLatt,SiteLong)
        clay.append(sc[0])
        sand.append(sc[1])
    



print("Finished calculating site parameters.")

# add column to the training set 
traingdataset["distancetoirport"]=distancetoirport
traingdataset["distancetolandfil"]=distancetolandfil
traingdataset["distancetochemfactory"]=distancetochemfactory
traingdataset["distancetofirefighting"]=distancetofirefighting
traingdataset["distancetowaterways"]=distancetowaterways

traingdataset["spilldata"]=spilldata
traingdataset["chemindustry"]=chemindustry

traingdataset["greenhouse"]=greenhouse
traingdataset["solidwaste"]=solidwaste
traingdataset["ustincident"]=ustincident

traingdataset["chemIndustowater"]=chemIndustowater
traingdataset["landfiltowater"]=landfiltowater
traingdataset["chemFacttowater"]=chemFacttowater


traingdataset["coalash"]=coalash
traingdataset["hazardwaste"]=hazardwaste
traingdataset["oldhazardsite"]=oldhazardsite

traingdataset["elevation"]=elevation

# add count of facilities within the positive sites 
traingdataset["countofChemIndus"]=countofChemIndus
traingdataset["countofLandfill"]=countofLandfill
traingdataset["countChemFactory"]=countChemFactory

traingdataset["solidwastetowater"]=solidwastetowater
traingdataset["countsolidwaste"]=countsolidwaste
traingdataset["industryname"]=industryname

traingdataset["ustincidenttowater"]=ustincidenttowater
traingdataset["counttoustincident"]=counttoustincident

traingdataset["firefightingtowater"]=ustincidenttowater
traingdataset["counttofirefighting"]=counttoustincident


traingdataset["distancetoprereglandfil"]=distancetoprereglandfil
traingdataset["prereglandfiltowater"]=prereglandfiltowater
traingdataset["counttofprereglandfil"]=counttofprereglandfil

traingdataset["fedsitetowater"]=counttofprereglandfil
traingdataset["counttoffedsites"]=counttofprereglandfil

if addsoil:
    traingdataset["soilclassification"]=soilclassification
    traingdataset["clay"]=clay
    traingdataset["sand"]=sand

traingdataset["temp"]=temp
traingdataset["Precipitation"]=Precipitation
traingdataset["windSpeed"]=windSpeed
traingdataset["Pressure"]=Pressure

traingdataset["distancetowaterdischarge"]=distancetowaterdischarge
traingdataset["dischargetowater"]=dischargetowater
traingdataset["countofwaterdischarge"]=countofwaterdischarge

traingdataset["landfillperp"]=landfillperp
traingdataset["prelandfillperp"]=prelandfillperp

traingdataset["WWTPflowrate"]=WWTPflowrate

traingdataset["distancetofirestation"]=distancetofirestation
traingdataset["firestationtowater"]=firestationtowater
traingdataset["countfirestation"]=countfirestation

#add water params 
traingdataset["landfillcountclosetowater"]=landfillcountclosetowater
traingdataset["distancetolandfillwater"]=distancetocheminduswater
traingdataset["cheminduscountclosetowater"]=cheminduscountclosetowater
traingdataset["distancetochemfacwater"]=distancetochemfacwater
traingdataset["chemfaccountclosetowater"]=chemfaccountclosetowater
traingdataset["distancetosolidwastewater"]=distancetosolidwastewater
traingdataset["solidwastecountclosetowater"]=solidwastecountclosetowater
traingdataset["distancetofirefightingwater"]=distancetofirefightingwater
traingdataset["firefightingcountclosetowater"]=firefightingcountclosetowater
traingdataset["distancetowaterdischargewater"]=distancetowaterdischargewater
traingdataset["waterdischargecountclosetowater"]=waterdischargecountclosetowater
traingdataset["septagecountclosetowater"]=septagecountclosetowater
traingdataset["distancetofirestationwater"]=distancetofirestationwater
traingdataset["firestationcountclosetowater"]=firestationcountclosetowater
traingdataset["distancetoustincidentwater"]=distancetoustincidentwater
traingdataset["ustincidentcountclosetowater"]=ustincidentcountclosetowater
traingdataset["distancetofedsitewater"]=distancetofedsitewater
traingdataset["fedsitecountclosetowater"]=fedsitecountclosetowater
 

#save the file 
traingdataset.to_csv('TrainingDataWithDistancenew2.csv', sep=',', encoding='utf-8', index=False)


#https://app.coolfarmtool.org/docs/api/v1/soil_grids/soil_grids.html





