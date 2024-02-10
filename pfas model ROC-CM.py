# Build model 

#import libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


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
from math import sin, cos, sqrt, atan2, radians
from sklearn.model_selection import KFold
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import completeness_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_pinball_loss, mean_squared_error
from numpy import set_printoptions
from sklearn.feature_selection import f_classif
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.ensemble import  RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from scipy.stats import pearsonr

import time
import warnings
import shap
warnings.filterwarnings("ignore")

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }


printdetails=True
#python -m pip install pyproj 

# set the font for charts and graphs  
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})

def reg_coef(x,y,label=None,hue=None,color=None,**kwargs):
    ax = plt.gca()
    r,p = pearsonr(x,y)
    ax.annotate('r = {:.2f}'.format(r), xy=(0.5,0.5), xycoords='axes fraction', ha='right')
    
 
    
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
         'landfillperp','prelandfillperp', 'WWTPflowrate','distancetofirestation','firestationtowater','countfirestation',
         'landfillcountclosetowater','distancetolandfillwater','cheminduscountclosetowater','distancetochemfacwater',
         'chemfaccountclosetowater','distancetosolidwastewater','solidwastecountclosetowater','distancetofirefightingwater',
         'firefightingcountclosetowater','distancetowaterdischargewater','waterdischargecountclosetowater','septagecountclosetowater',
         'distancetofirestationwater','firestationcountclosetowater','distancetoustincidentwater','ustincidentcountclosetowater',
         'distancetofedsitewater','fedsitecountclosetowater']
 

dataset = read_csv(url, names=names, header=0, index_col=False)

names = ['landfiltowater', 'Precipitation', 'distancetoirport','distancetolandfil',
        'distancetochemfactory','distancetofirefighting','distancetowaterways','spilldata',
        'chemindustry','greenhouse','solidwaste','ustincident','chemIndustowater','landfiltowater',
        'chemFacttowater','coalash','hazardwaste','oldhazardsite','elevation','countofChemIndus','countofLandfill',
        'countChemFactory','solidwastetowater','countsolidwaste','industryname','ustincidenttowater','counttoustincident',
        'firefightingtowater','counttofirefighting','distancetoprereglandfil','prereglandfiltowater','counttofprereglandfil',
        'fedsitetowater','counttoffedsites' ,'clay','sand' ,
        'temp','windSpeed','Pressure','distancetowaterdischarge','dischargetowater','countofwaterdischarge',
        'landfillperp','prelandfillperp','WWTPflowrate','distancetofirestation','firestationtowater','countfirestation',
        'landfillcountclosetowater','distancetolandfillwater','cheminduscountclosetowater','distancetochemfacwater',
        'chemfaccountclosetowater','distancetosolidwastewater','solidwastecountclosetowater','distancetofirefightingwater',
        'firefightingcountclosetowater','distancetowaterdischargewater','waterdischargecountclosetowater','septagecountclosetowater',
        'distancetofirestationwater','firestationcountclosetowater','distancetoustincidentwater','ustincidentcountclosetowater',
        'distancetofedsitewater','fedsitecountclosetowater' ,       
        'PFASFound' ] 


# encode the industry; type and soil classification
le = LabelEncoder()
industry = le.fit_transform(dataset['industryname'])
dataset['industryname']=industry

sitetype = le.fit_transform(dataset['Type'])
dataset['Type']=sitetype

soiltype = le.fit_transform(dataset['soilclassification'])
dataset['soilclassification']=soiltype


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


# remove high PFAS list from the data and 0 results 
#dataset.drop(dataset[dataset.PFAS ==0].index, inplace=True)
dataset.drop(dataset[dataset.PFAS > 300].index, inplace=True)

# set PFAS flag at 4 PPT as YES or NO 
dataset['PFASFound'] = np.where(dataset['PFAS'] == 0   , 0, 1)

# print the data set information
if printdetails:
    print(dataset.shape)
    print(dataset.info())
    print(dataset.describe())
    print(initial_eda(dataset))

 

coeflist=[]
if printdetails:
    for i in names:
        
        g = sns.lineplot(data=dataset, x='PFAS', y=i, markers=True, ci='sd' )
        #g = sns.pairplot(dataset, x_vars='PFAS', y_vars=i, kind='scatter',corner=False)
        coef = np.corrcoef(dataset['PFAS'], dataset[i])[0][1]
        coeflist.append(coef)
        # Make the label
        label = r'$\rho$ = ' + str(round(coef, 2))
        ax = plt.gca()
        ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)
        
        plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
        plt.rcParams['axes.titlepad'] = -14  # pad is in points...
        plt.title('Parameter = '+i, fontdict=font)
        plt.xlabel('Parts per Trillion (PPT)', fontdict=font)
        plt.ylabel('Correlation', fontdict=font)
        plt.show()

corrimp = pd.DataFrame()
corrimp["param"]=names
corrimp["coef"]=coeflist
corrimp.to_csv("coef.csv")


# drop the index 
dataset.drop('index',inplace=True, axis=1)


#drop unwanted columns and rows
dataset.drop('Type',inplace=True, axis=1)
dataset.drop('Latitude',inplace=True, axis=1)
dataset.drop('Longtitude',inplace=True, axis=1)
dataset.drop('Name',inplace=True, axis=1)
dataset.drop('soilclassification',inplace=True, axis=1)

#https://www.kaggle.com/code/prashant111/random-forest-classifier-feature-importance

# print the data set information
if printdetails:
    print(dataset.shape)
    print(dataset.info())
    print(dataset.describe())
    print(initial_eda(dataset))


# create instance of RFC
model = RandomForestClassifier(n_estimators= 1400, min_samples_split= 2, min_samples_leaf= 4, max_features='sqrt', max_depth=80, bootstrap=True)

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
        'fedsitetowater','counttoffedsites' ,'clay','sand' ,
        'temp','Precipitation','windSpeed','Pressure','distancetowaterdischarge','dischargetowater','countofwaterdischarge',
        'landfillperp','prelandfillperp','WWTPflowrate','distancetofirestation','firestationtowater','countfirestation',
        'landfillcountclosetowater','distancetolandfillwater','cheminduscountclosetowater','distancetochemfacwater',
        'chemfaccountclosetowater','distancetosolidwastewater','solidwastecountclosetowater','distancetofirefightingwater',
        'firefightingcountclosetowater','distancetowaterdischargewater','waterdischargecountclosetowater','septagecountclosetowater',
        'distancetofirestationwater','firestationcountclosetowater','distancetoustincidentwater','ustincidentcountclosetowater',
        'distancetofedsitewater','fedsitecountclosetowater',        
        'PFASFound' ] 

namesnp = np.array(names)

# add column to note PFAS based on the level and drop the actual PFAS value 
lim = 1
dataset.drop('PFAS',inplace=True, axis=1)



# reindex column to keep the PFAS Found column at the last 
dataset = dataset.reindex(columns=names)
print(dataset.head(20))
print(dataset.shape)
print(initial_eda(dataset))
 
# Split-out validation dataset
# We will split the loaded dataset into two, 80% of which we will use to train, 
# evaluate and select among our models, and 20% that we will hold back as a validation dataset.
array = dataset.values
X = array[:,0:len(names)-1]
y = array[:,len(names)-1]
rng = np.random.RandomState(0)


# split the data 
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=rng, shuffle=True)

#https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f
sel = SelectFromModel(RandomForestClassifier(n_estimators= 1400, min_samples_split= 2, min_samples_leaf= 4, max_features='sqrt', max_depth=80, bootstrap=True))
sel.fit(X_train, Y_train)
print(sel.get_support()) 


#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74


# Train the model 
model.fit(X_train, Y_train)

#https://mljar.com/blog/feature-importance-in-random-forest/


# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# parameter selection only once 


#calculate the time it takes to compute feature importance
#https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
start_time = time.time()
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

#now plot the feature importance
forest_importances = pd.Series(importances, index=namesnp[:-1])
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in Parameters")
plt.show()

fig.tight_layout()

# print score 
print(" Score : {} ".format(model.score(X_test, Y_test))) 



# predict the values
y_pred = model.predict(X_test)
y_true = Y_test

# create Confusion matrix
# https://github.com/tolgakurtulus/Machine/blob/master/confusion-matrix-with-random-forest.py
# https://www.youtube.com/watch?v=8Oog7TXHvFY
cm = confusion_matrix(y_true,y_pred)

# Plot the confusion matrix 
f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm, annot = True, linewidths = 0.5, linecolor = "red", fmt =".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Confusion Matrix model=RFC")
plt.show()

#plot ROC curve - “receiver operating characteristic
#define metrics
#instantiate the model
y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

# check kfold and print results         
kfold = StratifiedKFold(n_splits=10 ,random_state=0, shuffle=True)
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
print('%s: %f (%f)' % ("RFC", cv_results.mean(), cv_results.std()))
