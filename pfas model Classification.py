# Build model 

#import libraries 

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings


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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import completeness_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_pinball_loss, mean_squared_error
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from matplotlib import pyplot
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn import linear_model

 


warnings.filterwarnings("ignore")


printdetails=True
#python -m pip install meteostat 

 
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})

#function that presents initial EDA of dataset.
def initial_eda(df):
    if isinstance(df, pd.DataFrame):
        total_na = df.isna().sum().sum()
        print("Dimensions : %d rows, %d columns" % (df.shape[0], df.shape[1]))
        print("Total NA Values : %d " % (total_na))
        print("%38s %10s     %10s %10s" % ("Column Name", "Data Type", "#Distinct", "NA Values"))
        col_name = df.columns
        dtyp = df.dtypes
        uniq = df.nunique()
        na_val = df.isna().sum()
        for i in range(len(df.columns)):
            print("%38s %10s   %10s %10s" % (col_name[i], dtyp[i], uniq[i], na_val[i]))
        
    else:
        print("Expect a DataFrame but got a %15s" % (type(df)))

# Load dataset
url = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/TrainingDataWithDistancenew2.csv"
names = ['index','Type','Name','PFAS','Latitude','Longtitude','PFASFound','distancetoirport','distancetolandfil',
         'distancetochemfactory','distancetofirefighting','distancetowaterways','spilldata',
         'chemindustry','greenhouse','solidwaste','ustincident','chemIndustowater','landfiltowater',
         'chemFacttowater','coalash','hazardwaste','oldhazardsite','elevation','countofChemIndus','countofLandfill',
         'countChemFactory','solidwastetowater','countsolidwaste','industryname','ustincidenttowater','counttoustincident',
         'firefightingtowater','counttofirefighting','distancetoprereglandfil','prereglandfiltowater','counttofprereglandfil',
         'fedsitetowater','counttoffedsites', 'soilclassification','clay','sand',
         'temp','Precipitation','windSpeed','Pressure','distancetowaterdischarge','dischargetowater','countofwaterdischarge',
         'landfillperp','prelandfillperp','WWTPflowrate','distancetofirestation','firestationtowater','countfirestation',
         'landfillcountclosetowater','distancetolandfillwater','cheminduscountclosetowater','distancetochemfacwater',
         'chemfaccountclosetowater','distancetosolidwastewater','solidwastecountclosetowater','distancetofirefightingwater',
         'firefightingcountclosetowater','distancetowaterdischargewater','waterdischargecountclosetowater','septagecountclosetowater',
         'distancetofirestationwater','firestationcountclosetowater','distancetoustincidentwater','ustincidentcountclosetowater',
         'distancetofedsitewater','fedsitecountclosetowater']

dataset = read_csv(url, names=names, header=0, index_col=False)

# find the average for the sand and clay
claymean = (dataset['clay'].mean())
sandmean = (dataset['sand'].mean())
tempmean= (dataset['temp'].mean())
precmean= (dataset['Precipitation'].mean())
wnspdmean= (dataset['windSpeed'].mean())
presmean= (dataset['Pressure'].mean())

# replacce nan values with average
dataset['clay'] = dataset['clay'].replace(np.nan, claymean)
dataset['sand'] = dataset['sand'].replace(np.nan, sandmean)
dataset['temp'] = dataset['temp'].replace(np.nan, tempmean)
dataset['Precipitation'] = dataset['sand'].replace(np.nan, precmean)
dataset['windSpeed'] = dataset['sand'].replace(np.nan, wnspdmean)
dataset['Pressure'] = dataset['sand'].replace(np.nan, presmean)


dataset['landfillperp'] = dataset['landfillperp'].replace(np.nan, wnspdmean)
dataset['prelandfillperp'] = dataset['prelandfillperp'].replace(np.nan, presmean)
dataset['WWTPflowrate'] = dataset['WWTPflowrate'].replace(np.nan, presmean)



# encode the industry 
le = LabelEncoder()
industry = le.fit_transform(dataset['industryname'])
dataset['industryname']=industry

sitetype = le.fit_transform(dataset['Type'])
dataset['Type']=sitetype

soiltype = le.fit_transform(dataset['soilclassification'])
dataset['soilclassification']=soiltype


#drop unwanted columns and rows
#remover header 
# dataset.drop(index=dataset.index[0], axis=0, inplace=True)

dataset.drop('index',inplace=True, axis=1)
dataset.drop('Type',inplace=True, axis=1)
dataset.drop('Latitude',inplace=True, axis=1)
dataset.drop('Longtitude',inplace=True, axis=1)
dataset.drop('Name',inplace=True, axis=1)

dataset.drop(dataset[dataset.PFAS > 200].index, inplace=True)

# remove high PFAS list from the data
#dataset.drop(dataset[dataset.PFAS >= 200].index, inplace=True)

#https://www.kaggle.com/code/prashant111/random-forest-classifier-feature-importance
if printdetails:
    print(dataset.shape)
    print(dataset.info())
    print(dataset.describe())
    print(initial_eda(dataset))

#df.describe(include='all')
# check if there is any null value
#print(dataset.isnull().sum())

# add a column to flag if PFAS is found or not
PFASlimit = [0,1,3,5,10,20,30, 40, 50, 60, 70,80, 90, 100, 125]

# Spot Check Algorithms
models = []

models.append(('SVM', SVC(gamma='auto', kernel='rbf')))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('CART', DecisionTreeClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('GNB', GaussianNB()))
models.append(('RFC', RandomForestClassifier(n_estimators= 1400, min_samples_split= 2, min_samples_leaf= 4, max_features='sqrt', max_depth=80, bootstrap=True)))
models.append(('SVM3',SVC(kernel='sigmoid', gamma="auto")))
#models.append(('SVM4',SVC(kernel='linear', gamma="auto")))
 

masterdata = dataset
resultsfromtrial= []
featureimportancelist=[]

# rearrgange the columns 
names = [ 'distancetoirport','distancetolandfil',
        'distancetochemfactory','distancetofirefighting','distancetowaterways','spilldata',
        'chemindustry','greenhouse','solidwaste','ustincident','chemIndustowater','landfiltowater',
        'chemFacttowater','coalash','hazardwaste','oldhazardsite','elevation','countofChemIndus','countofLandfill',
        'countChemFactory','solidwastetowater','countsolidwaste','industryname','ustincidenttowater','counttoustincident',
        'firefightingtowater','counttofirefighting','distancetoprereglandfil','prereglandfiltowater','counttofprereglandfil',
        'fedsitetowater','counttoffedsites','soilclassification','clay','sand' ,
        'temp','Precipitation','windSpeed','Pressure','distancetowaterdischarge','dischargetowater','countofwaterdischarge',
        'landfillperp','prelandfillperp','WWTPflowrate','distancetofirestation','firestationtowater','countfirestation',
        'landfillcountclosetowater','distancetolandfillwater','cheminduscountclosetowater','distancetochemfacwater',
        'chemfaccountclosetowater','distancetosolidwastewater','solidwastecountclosetowater','distancetofirefightingwater',
        'firefightingcountclosetowater','distancetowaterdischargewater','waterdischargecountclosetowater','septagecountclosetowater',
        'distancetofirestationwater','firestationcountclosetowater','distancetoustincidentwater','ustincidentcountclosetowater',
        'distancetofedsitewater','fedsitecountclosetowater',
        'PFASFound' ] 

featurelist=[ 'distancetoirport','distancetolandfil',
        'distancetochemfactory','distancetofirefighting','distancetowaterways','spilldata',
        'chemindustry','greenhouse','solidwaste','ustincident','chemIndustowater','landfiltowater',
        'chemFacttowater','coalash','hazardwaste','oldhazardsite','elevation','countofChemIndus','countofLandfill',
        'countChemFactory','solidwastetowater','countsolidwaste','industryname','ustincidenttowater','counttoustincident',
        'firefightingtowater','counttofirefighting','distancetoprereglandfil','prereglandfiltowater','counttofprereglandfil',
        'fedsitetowater','counttoffedsites' , 'soilclassification','clay','sand' ,
        'temp','Precipitation','windSpeed','Pressure', 'distancetowaterdischarge','dischargetowater','countofwaterdischarge',
        'landfillperp','prelandfillperp','WWTPflowrate','distancetofirestation','firestationtowater','countfirestation',
        'landfillcountclosetowater','distancetolandfillwater','cheminduscountclosetowater','distancetochemfacwater',
        'chemfaccountclosetowater','distancetosolidwastewater','solidwastecountclosetowater','distancetofirefightingwater',
        'firefightingcountclosetowater','distancetowaterdischargewater','waterdischargecountclosetowater','septagecountclosetowater',
        'distancetofirestationwater','firestationcountclosetowater','distancetoustincidentwater','ustincidentcountclosetowater',
        'distancetofedsitewater','fedsitecountclosetowater']

featureimportancelist.append(featurelist)

for lim in PFASlimit:
    
    testdataset = dataset.copy()

    print("fitting model for PFAS threshold ",lim)
    #set various limit to see model performance
    testdataset['PFASFound'] = np.where(testdataset['PFAS'] > lim, 1, 0)
    testdataset.drop('PFAS',inplace=True, axis=1)

    testdataset = testdataset.reindex(columns=names)

    # Split-out validation dataset
    # We will split the loaded dataset into two, 80% of which we will use to train, 
    # evaluate and select among our models, and 20% that we will hold back as a validation dataset.
    array = testdataset.values
    X = array[:,0:len(names)-1]
    y = array[:,len(names)-1]
    rng = np.random.RandomState(0)

    # split the data 
    #https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=rng, shuffle=True)

    # [ True False False False False False  True  True  True  True False False
    # False False False False False False  True  True False  True False  True
    #  False False False False False  True False False]

    #https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    #https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

    results = []
    mnames=[]
    # evaluate each model in turn
    for name, model in models:    
            model.fit(X_train, Y_train)

            #print(testacc, trainacc)
            # Compare Algorithms
            #print('%s: %f %f' % (name, model.score(X_train, Y_train)*100 , model.score(X_test, Y_test)*100 ))
            resultsfromtrial.append ([lim,name, model.score(X_train, Y_train)*100 , model.score(X_test, Y_test)*100])
       
            
            kfold = StratifiedKFold(n_splits=10 ,random_state=0, shuffle=True)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
            results.append(cv_results*100)
            mnames.append(name)
            print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
            
            if (name =='RFC'):
                #print(model.feature_importances_ )
                featureimportancelist.append(model.feature_importances_)

    
    plt.boxplot(results, labels=mnames)
    plt.title('Algorithm Comparison: threshold for PFAS level = '+str(lim)+' PPT')
    plt.xlabel('Machine Learning Models')
    plt.ylabel('Model Accuracy %')
    #plt.show()

resultdata = pd.DataFrame(resultsfromtrial)
resultdata.to_csv("result.csv")

featuredata = pd.DataFrame(featureimportancelist)
featuredata.to_csv("feature.csv")


print(resultdata)


