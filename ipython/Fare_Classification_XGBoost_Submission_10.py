import numpy as np
import pandas as pd

train_df = pd.read_csv('../train.csv',index_col="tripid")

ax = pd.DataFrame({'not null': train_df.count(),
                   'null': train_df.isnull().sum()}).plot.barh(stacked=True)

ax.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5)
)

train_df['label'].value_counts().plot.barh()

from sklearn.preprocessing import label_binarize
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler

RANDOM_SEED = 42  

labels_raw_df = train_df.loc[:, train_df.columns == 'label']
features_raw_df = train_df.drop(columns=['label'])


def preprocess_labels(input_labels_df):
  labels_df = input_labels_df
  labels_df = pd.DataFrame(label_binarize(labels_df, classes=['incorrrect', 'correct']))
  
  labels_df.columns = labels_df.columns
  labels_df.index = labels_df.index

  return labels_df  

labels_df = preprocess_labels(labels_raw_df)

def preprocess_features(input_features_df):
    features_df = input_features_df.copy()
    
    #date time
    features_df['pickup_time'] = pd.to_datetime(features_df['pickup_time'])
    features_df['drop_time'] = pd.to_datetime(features_df['drop_time'])
    
    #duration
    features_df['duration_fill'] = (features_df['drop_time'] - features_df['pickup_time']).dt.total_seconds()
    features_df['duration'] = features_df['duration'].fillna(features_df['duration_fill'])
    
    features_df = features_df.drop(columns=['duration_fill'])

    #travel time
    features_df['travel_time'] = features_df['duration'] - features_df['meter_waiting']
    
    #waiting time
    features_df['waiting_time'] = features_df['meter_waiting'] + features_df['meter_waiting_till_pickup']
    
    #distance fare
    features_df['distance_fare'] = features_df['fare'] - features_df['meter_waiting_fare'] - features_df['additional_fare']
   
    #distance
    features_df['manhattan_distance'] = (features_df['drop_'] - features_df['pickup_time'])

    #get day of the week
    features_df['pickup_day_of_week'] = features_df['pickup_time'].dt.day_name()
    features_df['drop_day_of_week'] = features_df['drop_time'].dt.day_name()
    
    features_df['pickup_hour_float'] = features_df['pickup_time'].dt.hour + features_df['pickup_time'].dt.minute/60
    features_df['drop_hour_float'] = features_df['drop_time'].dt.hour + features_df['drop_time'].dt.minute/60

    
    #encode cyclic 24 hours with sin and cos
    features_df['sin_pickup_time'] = np.sin(2*np.pi*features_df.pickup_hour_float/24.)
    features_df['cos_pickup_time'] = np.cos(2*np.pi*features_df.pickup_hour_float/24.)

    features_df['sin_drop_time'] = np.sin(2*np.pi*features_df.drop_hour_float/24.)
    features_df['cos_drop_time'] = np.cos(2*np.pi*features_df.drop_hour_float/24.)
    
    features_df = pd.get_dummies(features_df, columns = ['pickup_day_of_week', 'drop_day_of_week'])
    
    features_df = features_df.drop(columns=['pickup_time','drop_time', 'pickup_hour_float', 'drop_hour_float'])
    
    numeric_cols = features_df.columns[features_df.dtypes != "object"].values
    
    non_numeric_cols = features_df.columns[features_df.dtypes == "object"].values
    
    #numeric
    scaler = MinMaxScaler()
    features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])

    imputer = SimpleImputer(strategy='mean')
    features_df[numeric_cols] = imputer.fit_transform(features_df[numeric_cols])
    
    return features_df

features_df = preprocess_features(features_raw_df)

test_features_raw_df = pd.read_csv('../test.csv',index_col="tripid")
test_features_df = preprocess_features(test_features_raw_df)

from sklearn.model_selection import train_test_split

X_train, X_eval, y_train, y_eval = train_test_split(
    features_df,
    labels_df,
    test_size=0.2,
    shuffle=True,
    stratify=labels_df,
    random_state=RANDOM_SEED
)

import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_eval, label=y_eval)

from xgboost.sklearn import XGBClassifier
from sklearn.metrics import f1_score

def f1(y_pred, dtrain):
  print(dtrain)
  y_true = dtrain.get_label()
  err = 1 - f1_score(y_true, np.round(y_pred))
  return 'f1', err

from bayes_opt import BayesianOptimization

def bo_tune_xgb(max_depth, min_split_loss, n_estimators ,learning_rate, subsample, min_child_weight):
    params = {
    'learning_rate': learning_rate, 
    'min_split_loss': min_split_loss,
    'n_estimators': int(n_estimators), 
    'objective': 'binary:logistic', 
    'max_depth': int(max_depth), 
    'min_child_weight': min_child_weight,
    'subsample': subsample,
    'num_parallel_tree' : 10,
    'nthread' : -1}

    cv_result = xgb.cv(
                        params,
                        dtrain,
                        seed=42,
                        nfold=5,
                        feval = f1,
                        early_stopping_rounds=10,
                        verbose_eval=0
                    )
    
    return -1.0 * cv_result['test-f1-mean'].iloc[-1]

xgb_bo = BayesianOptimization(bo_tune_xgb, {'max_depth': (6, 14),
                                            'min_split_loss': (0, 1),
                                            'learning_rate':(0,2),
                                            'n_estimators':(50,100),
                                            'subsample' :(0.5,1),
                                            'min_child_weight' : (0,2)
                                            })

xgb_bo.maximize(init_points=30, n_iter=50, acq='ei', xi=0.0)

with open('log.log', 'a') as logfile:
    logfile.write(f'{xgb_bo.max}')

params = {
    'learning_rate': 0.2, 
    'min_split_loss': 0.2,
    'n_estimators': 100, 
    'objective': 'binary:logistic', 
    'max_depth': 6, 
    'min_child_weight': 2, 
    'max_delta_step': 0, 
    'subsample': 1,
    'num_parallel_tree' : 5,
    'nthread' : -1}

for k in (xgb_bo.max['params']):
    params[k] = xgb_bo.max['params'][k]

params['max_depth']= int(params['max_depth'])
params['n_estimators']= int(params['n_estimators'])

with open('log.log', 'a') as logfile:
    logfile.write(f'{params}')

cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=1000,
    seed=42,
    nfold=5,
    feval = f1,
    early_stopping_rounds=10,
    verbose_eval=0
)

print(f"F1 Score: {1-cv_results['test-f1-mean'].min()}")

model = XGBClassifier(**params).fit(X_train, y_train.values.ravel())

y_pred = model.predict(X_eval)

thresh = .5
y_pred [y_pred > thresh] = 1
y_pred [y_pred <= thresh] = 0

from sklearn.metrics import f1_score

print('F1 Score = {:.6f}'.format(f1_score(y_eval, y_pred)))
print('Macro F1 Score = {:.6f}'.format(f1_score(y_eval, y_pred, average='macro')))

feature_importnace = model.get_score(importance_type='gain')

import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.bar(range(len(feature_importnace)), list(feature_importnace.values()), align='center')
plt.xticks(range(len(feature_importnace)), list(feature_importnace.keys()), rotation='vertical')

plt.show

def predict_test_set(model, test_features_df, submission_no):
    global thresh
    
    test_pred = model.predict(xgb.DMatrix(test_features_df))
    
    test_pred [test_pred > thresh] = 1
    test_pred [test_pred <= thresh] = 0
    
    submission_df = pd.read_csv('../sample_submission.csv',index_col="tripid")
    
    # Make sure we have the rows in the same order
    np.testing.assert_array_equal(test_features_df.index.values, submission_df.index.values)
    
    # Save predictions to submission data frame
    submission_df["prediction"] = test_pred
    
    submission_df['prediction'] = submission_df['prediction'].astype(int)

    submission_df.to_csv(f'../submission{submission_no}.csv', index=True)


predict_test_set(model, test_features_df, 10)
