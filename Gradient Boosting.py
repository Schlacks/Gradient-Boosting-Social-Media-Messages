from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
import time
import pandas as pd
import seaborn as sns
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import  train_test_split
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

df=pd.read_csv('OnlineNewsPopularity.csv')
start_time = time.time()
# A random sample of 18% of the data is chosen, to keep computing time realistic (about an hour). This can be adjusted at will. For quick demonstration
# purposes this value should not be bigger than 1%.
r=np.random.choice(range(len(df)),round(0.18*len(df)),replace=False)
dfshort=df.iloc[(r)]

X=dfshort.drop([' shares','url'],1)
y=dfshort[' shares']

#Checking for missing values
pd.isnull(X).sum()

#Split the data set into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# A grid of parameters is set up. And all combinations of these are simulated. This may take a while (~1h) since 4*2*4*3=96 models are fit.
param_grid={'learning_rate':[0.1, 0.05, 0.02,0.01],
            'max_depth':[4,6],
            'min_samples_leaf':[3,5,9,17],
            'max_features':[1.0,0.3,0.1]}

est = GradientBoostingRegressor(n_estimators=500)
gs_cv=GridSearchCV(est,param_grid).fit(X_train,y_train)

gs_cv.best_params_


#The optimal number of estimates is searched to avoid overfitting, using the previously estimated values
nr_of_est=np.linspace(2,1600,15)
mean_MAE_tot=[]
yerr=[]
MAE_tot=[]
for i1 in nr_of_est:
    for i in range(3):
        est = GradientBoostingRegressor(n_estimators=int(i1), learning_rate=gs_cv.best_params_['learning_rate'], max_depth=gs_cv.best_params_['max_depth'],
                                        max_features=gs_cv.best_params_['max_features'],min_samples_leaf=gs_cv.best_params_['min_samples_leaf'])
        model = est.fit(X_train, y_train)
        pred = model.predict(X_test)

        MAE = sum(abs(pred - y_test)) / len(pred)
        MAE_tot.append(MAE)
    mean_acc = (sum(MAE_tot) / len(MAE_tot))
    standard_error = np.std(MAE_tot) / sqrt(len(MAE_tot))
    mean_MAE_tot.append(mean_acc)
    yerr.append(standard_error)
#The search is plotted and the number of esitimators that resulted in the least Mean Absolute Error is chosen
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.errorbar(nr_of_est,mean_MAE_tot,yerr=yerr,color='b')
ax.set_title('Effect of Number of Estimation on Error')
ax.set_ylabel('Mean Absolut Error')
ax.set_xlabel('Number of Estimations')
n_estimators=nr_of_est[np.argmin(mean_MAE_tot)]


#Next, the value for the learning rate epsilon is fine tuned. Values in increments of 10% of the previously estimated learning rate are tested ranging
#from +50% to -50%
mean_MAE_tot=[]
yerr=[]
epsilon_range=[gs_cv.best_params_['learning_rate']+x*gs_cv.best_params_['learning_rate'] for x in np.linspace(-.5,.5,11)]
for i in epsilon_range:
    for i1 in range(10):
        est = GradientBoostingRegressor(n_estimators=int(n_estimators), learning_rate=float(i),
                                        max_depth=gs_cv.best_params_['max_depth'],
                                        max_features=gs_cv.best_params_['max_features'],
                                        min_samples_leaf=gs_cv.best_params_['min_samples_leaf'])
        model = est.fit(X_train, y_train)
        pred = model.predict(X_test)
        MAE = sum(abs(pred - y_test)) / len(pred)
        MAE_tot.append(MAE)
    mean_acc = (sum(MAE_tot) / len(MAE_tot))
    standard_error = np.std(MAE_tot) / sqrt(len(MAE_tot))
    mean_MAE_tot.append(mean_acc)
    yerr.append(standard_error)

#Again,this search is visualized
f, ax = plt.subplots()
ax.errorbar(epsilon_range,mean_MAE_tot,yerr=yerr,color='b')
ax.set_title('Effect of Learning Rate on Error')
ax.set_ylabel('Mean Absolut Error')
ax.set_xlabel('Learningrate Epsilon')
final_learningrate=epsilon_range[np.argmin(mean_MAE_tot)]


#One last single model is fit with all the previously estimated parameters
est = GradientBoostingRegressor(n_estimators=int(n_estimators), learning_rate=final_learningrate,
                                max_depth=gs_cv.best_params_['max_depth'],
                                max_features=gs_cv.best_params_['max_features'],
                                min_samples_leaf=gs_cv.best_params_['min_samples_leaf'])
model = est.fit(X_train, y_train)

#This model is used to plot the Feature Importance
imp=model.feature_importances_
imp1=pd.DataFrame(imp,index=X_train.columns)
imp1.columns=['estimator_importance']
imp1=imp1.sort(columns='estimator_importance')
fig, ax = plt.subplots()
ax.barh(np.arange(len(X_train.columns)), imp1['estimator_importance'], color='b')
ax.set_yticks(np.arange(len(X_train.columns)))
ax.set_yticklabels(imp1.index.values)
ax.set_title('Feature Importance')


model=est.fit(X_train,y_train)
pred=model.predict(X_test)
MAE=sum(abs(pred-y_test))/len(pred)
print('This model achieves a Mean Absolute Error of:',MAE,'.')
print('Its pramaters are:')
print('Number of estimators (i.e. Numbers of trees fitted):',n_estimators)
print('Learning rate:',final_learningrate)
print('Max depth (i.e. maximum number of splits per estimation:',gs_cv.best_params_['max_features'])
print('Min samples leaf (i.e. minimum number of samples to be in a leaf):',gs_cv.best_params_['min_samples_leaf'])


regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
pred=regr.predict(X_test)
MAE=sum(abs(pred-y_test))/len(pred)

print('The MAE of the current is considerably smaller than the MAE achieved through Ordinary Least Square Regression. The MAE of OLS on the same data is:', MAE)

print("--- %s seconds ---" % (time.time() - start_time))