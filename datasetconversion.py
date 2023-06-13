import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
dataset1 = pd.read_csv('dataset1.csv')
dataset1 = dataset1.drop('customerID',axis=1)
dataset1['TotalCharges'] = pd.to_numeric(dataset1['TotalCharges'],errors='coerce')
dataset1.drop(dataset1[dataset1['TotalCharges'].isnull()].index,inplace=True)
dataset1.reset_index(drop=True,inplace=True)
dataset1.replace('No internet service', 'No', inplace=True)
dataset1.replace('No phone service', 'No', inplace=True)
dataset1['gender'].replace({'Female':1,'Male':0},inplace=True)
more_than_2 = ['InternetService' ,'Contract' ,'PaymentMethod']
dataset1 = pd.get_dummies(data=dataset1, columns= more_than_2)
two_cate = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
for i in two_cate:
    dataset1[i].replace({"No":0, "Yes":1}, inplace=True)
scaler = StandardScaler()
large_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
dataset1[large_cols] = scaler.fit_transform(dataset1[large_cols])
dataset1.to_csv('dataset1_processed.csv')


dataset2 = pd.read_csv('trainmini.csv')
print(dataset2)
print(dataset2.isnull().sum())
print(dataset2.dtypes)
dataset2['churn'].replace({"yes":1,"no":0},inplace=True)
print(dataset2.dtypes)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
def FunLabelEncoder(df):
    for c in df.columns:
        if df.dtypes[c] == object:
            le.fit(df[c].astype(str))
            df[c] = le.transform(df[c].astype(str))
    return df
dataset2 = FunLabelEncoder(dataset2)
print(dataset2.dtypes)
print(dataset2)

numeric = ['number_vmail_messages', 'total_day_minutes',
       'total_day_calls', 'total_day_charge', 'total_eve_minutes',
       'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
       'total_night_calls', 'total_night_charge', 'total_intl_minutes',
       'total_intl_calls', 'total_intl_charge',
       'number_customer_service_calls']
dataset2[numeric] = scaler.fit_transform(dataset2[numeric])
print(dataset2)
dataset2.to_csv('dataset2_processed.csv')


dataset3 = pd.read_csv('dataset3train.csv')
Churn = {'Yes':1,'No':0}
dataset3.Churn = [Churn[item] for item in dataset3.Churn]
dataset3['MonthlyRevenue'] = dataset3['MonthlyRevenue'].replace(np.nan, dataset3['MonthlyRevenue'].mean())
dataset3['MonthlyMinutes'] = dataset3['MonthlyMinutes'].replace(np.nan, dataset3['MonthlyMinutes'].mean())
dataset3['TotalRecurringCharge'] = dataset3['TotalRecurringCharge'].replace(np.nan, dataset3['TotalRecurringCharge'].mean())
dataset3['OverageMinutes'] = dataset3['OverageMinutes'].replace(np.nan, dataset3['OverageMinutes'].mean())
dataset3['DirectorAssistedCalls'] = dataset3['DirectorAssistedCalls'].replace(np.nan, dataset3['DirectorAssistedCalls'].mean())
dataset3['RoamingCalls'] = dataset3['RoamingCalls'].replace(np.nan, dataset3['RoamingCalls'].mean())
dataset3['PercChangeMinutes'] = dataset3['PercChangeMinutes'].replace(np.nan, dataset3['PercChangeMinutes'].mean())
dataset3['PercChangeRevenues'] = dataset3['PercChangeRevenues'].replace(np.nan, dataset3['PercChangeRevenues'].mean())
dataset3.drop(dataset3[dataset3['ServiceArea'].isnull()].index, inplace=True)
dataset3.reset_index(drop=True, inplace=True)
dataset3.drop(dataset3[dataset3['Handsets'].isnull()].index, inplace=True)
dataset3.reset_index(drop=True, inplace=True)
dataset3.drop(dataset3[dataset3['HandsetModels'].isnull()].index, inplace=True)
dataset3.reset_index(drop=True, inplace=True)
dataset3['AgeHH1'] = dataset3['AgeHH1'].replace(np.nan,dataset3['AgeHH1'].mean())
dataset3['AgeHH2'] = dataset3['AgeHH2'].replace(np.nan,dataset3['AgeHH2'].mean())
dataset3 = dataset3.drop('CustomerID',axis=1)
dataset3 = FunLabelEncoder(dataset3)
print(dataset3.info())

numeric = ['MonthlyRevenue','MonthlyMinutes','TotalRecurringCharge','DirectorAssistedCalls',
'OverageMinutes','RoamingCalls','PercChangeMinutes','PercChangeRevenues','DroppedCalls',
'BlockedCalls','UnansweredCalls','CustomerCareCalls','ThreewayCalls','ReceivedCalls',
'OutboundCalls','InboundCalls','PeakCallsInOut','OffPeakCallsInOut','DroppedBlockedCalls',
'CallForwardingCalls','CallWaitingCalls','Handsets','HandsetModels','CurrentEquipmentDays',
'AgeHH1','AgeHH2']
dataset3[numeric] = scaler.fit_transform(dataset3[numeric])
data = dataset3[['MonthlyMinutes','TotalRecurringCharge','PercChangeMinutes','UnansweredCalls',
                  'CustomerCareCalls','ReceivedCalls','OutboundCalls','InboundCalls',
                  'PeakCallsInOut','OffPeakCallsInOut','UniqueSubs','Handsets',
                  'HandsetModels','CurrentEquipmentDays','AgeHH1','HandsetWebCapable',
                  'RetentionCalls','RetentionOffersAccepted','HandsetPrice',
                  'MadeCallToRetentionTeam','CreditRating','Churn']]
data.to_csv('dataset3_processed.csv')