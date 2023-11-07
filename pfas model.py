# Build model 

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
import warnings

warnings.filterwarnings("ignore")

#py -m pip install xgboost 

# Load dataset
url = "https://raw.githubusercontent.com/aadarsh0709/pfasdatafiles/main/TrainingDataWithDistancenew2.csv"
names = ['index','Site Type','Name','Total PFAS','Latitude','Longtitude','distancetoirport','distancetolandfil','distancetochemfactory',
         'distancetofirefighting','distancetowaterways','spilldata','chemindustry','federalsites','greenhouse','solidwaste','ustincident',
         'chemtowater','landfiltowater','coalash','hazardwaste','oldhazardsite','elevation']

dataset = read_csv(url, names=names, header=None, index_col=False)

#drop unwanted columns and rows
dataset.drop(index=dataset.index[0], axis=0, inplace=True)
dataset.drop('index',inplace=True, axis=1)
dataset.drop('Name',inplace=True, axis=1)
dataset.drop('Latitude',inplace=True, axis=1)
dataset.drop('Longtitude',inplace=True, axis=1)

# transform site type
mapping_dict = {'Waterways': 1, 
				'Drinking Water': 2,
                'Military Site': 3,
                'Well': 4
			     }

dataset['Site Type'] = dataset['Site Type'].map(mapping_dict)



# rearrgange the columns 
new_cols = ['Site Type','distancetoirport','distancetolandfil','distancetochemfactory',
         'distancetofirefighting','distancetowaterways','spilldata','chemindustry','federalsites','greenhouse','solidwaste','ustincident',
         'chemtowater','landfiltowater','coalash','hazardwaste','oldhazardsite','elevation','Total PFAS']
dataset = dataset.reindex(columns=new_cols)

#dataset.drop(new_cols[0],inplace=True, axis=1)
dataset.drop(['distancetoirport'],inplace=True, axis=1)
dataset.drop(['distancetochemfactory'],inplace=True, axis=1)
dataset.drop(['federalsites'],inplace=True, axis=1)

...
# Split-out validation dataset
# We will split the loaded dataset into two, 80% of which we will use to train, 
# evaluate and select among our models, and 20% that we will hold back as a validation dataset.
array = dataset.values
X = array[:,0:14]
y = array[:,14]
rng = np.random.RandomState(0)

# split the data 
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=rng, shuffle=True)

# Spot Check Algorithms
models = []
models.append(('SVM', SVC(gamma='auto', kernel='rbf')))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('CART', DecisionTreeClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('GNB', GaussianNB()))
models.append(('RFC', RandomForestClassifier(n_estimators= 250, min_samples_split= 2, min_samples_leaf= 1, max_features='sqrt', max_depth=50, bootstrap=True)))


results = []
names = []
# evaluate each model in turn
for name, model in models:
        
        model.fit(X_train, Y_train)
        # Compare Algorithms
        print('%s: %f %f' % (name, model.score(X_train, Y_train)*100 , model.score(X_test, Y_test)*100 ))

        kfold = StratifiedKFold(n_splits=10 ,random_state=0, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        #print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#plt.boxplot(results, labels=names)
#plt.title('Algorithm Comparison')
#plt.show()


