# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:00:13 2023

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_squared_error, mean_absolute_error,accuracy_score,r2_score
from tqdm import tqdm_notebook as tqdm
from sklearn.ensemble import RandomForestRegressor
from prophet.plot import plot_plotly, plot_components_plotly
import warnings
warnings.filterwarnings('ignore')

def fill_nans(df, columns_to_fill):
    for col in columns_to_fill:
        df[col].fillna(method='ffill', inplace=True)
    return df

def fill_nans2(df, columns_to_fill):
    for col in columns_to_fill:
        df[col].fillna(method='bfill', inplace=True)
    return df

#Old Method
df = pd.read_csv("Path to inp file")
vr_cols = ['VendorId','VendorName','RetailerID','Retailer','StoreID','ProductID','ActualSaleDate']
vr = pd.read_csv("path to vendor_info",names=vr_cols)
vr['ActualSaleDate'] = pd.to_datetime(vr['ActualSaleDate'])
ans1 = vr.groupby(['ActualSaleDate','StoreID','ProductID'])['VendorId', 'RetailerID'].nunique().reset_index()
ans1_fin = ans1[ans1['VendorId']<2].drop(columns = ['VendorId', 'RetailerID'])
vr_fin = ans1_fin.merge(vr,on = ['ActualSaleDate', 'StoreID', 'ProductID'],how = 'left')
#chk_vr = vr[(vr['StoreID'] == 152897) & (vr['ProductID'] == 41582)]

df['ActualSaleDate'] = pd.to_datetime(df['ActualSaleDate'])
df = df.merge(vr_fin,on=['StoreID','ProductID','ActualSaleDate'],how = 'left')
#df = df.rename(columns = {"VendorId":"VendorID","Retailer":"RetailerName"})

#New method
'''#df = pd.read_csv("path to inp file")
df = pd.read_csv("patu to file")
df = df[['ActualSaleDate', 'StoreID', 'ProductID', 'str_sku_id', 'ProductName','RetailPrice', 'storeCity', 'storeState', 'StoreZip5', 'Inv_morn','Inv_eod', 'QtySold', 'unitssolds_in_zip', 'tavg', 'prcp', 'wspd','snow', 'proportion_sale', 'VendorID', 'VendorName', 'RetailerID','RetailerName']]
df = df.rename(columns = {'RetailerName':'Retailer','VendorID':'VendorId'})
df['ActualSaleDate'] = pd.to_datetime(df['ActualSaleDate'])'''

'''end_date = pd.to_datetime('2022-06-30')
df['Rec_type'] = np.where(df['ActualSaleDate']>end_date,"Test","Train")

df_train = df[df['Rec_type']=='Train']
train_cts = pd.DataFrame(df_train.groupby(['str_sku_id'])['Rec_type'].count()).rename(columns = {'Rec_type':'Train_cts'}).reset_index()
df_test = df[df['Rec_type']=='Test']
test_cts = pd.DataFrame(df_test.groupby(['str_sku_id'])['Rec_type'].count()).rename(columns = {'Rec_type':'Test_cts'}).reset_index()
train_cts = train_cts.merge(test_cts,on=['str_sku_id'],how = 'left')
train_cts['chk'] = np.where(train_cts['Train_cts']>train_cts['Test_cts'],"Yes","No")
train_cts['chk'].value_counts()'''

df['Day_of_week'] = df['ActualSaleDate'].dt.strftime('%A')
cal = calendar()
holidays = cal.holidays(start='2021-01-01', end='2022-12-31')
df['Holiday'] = df['ActualSaleDate'].isin(holidays)
df['Holiday'] = df['Holiday'].astype(int)

def map_to_season(date):
    if date.month in [3, 4, 5]:
        return 'Spring'
    elif date.month in [6, 7, 8]:
        return 'Summer'
    elif date.month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

# Apply the function to create a 'season' column
df['Season'] = df['ActualSaleDate'].apply(map_to_season)
#rd = pd.read_csv("path to vendor route")
#rd = rd[['VendorID', 'StoreID', 'ProductID', 'RouteID','VendorName']].drop_duplicates().rename(columns = {'VendorID':'VendorId'})
#chk_rd = rd[(rd['StoreID']==9310)&(rd['ProductID']==45616)]
#df1 = chk.merge(rd,on=['VendorId','StoreID', 'ProductID'],how = 'left')
df_encoded = pd.get_dummies(df, columns=['Holiday', 'Day_of_week','Season'],prefix=['Holiday', 'Day_of_week','Season'], prefix_sep='_')
str_sku = df_encoded['str_sku_id'].drop_duplicates()

end_date = pd.to_datetime('2022-06-30')
chk = df_encoded[(df_encoded['StoreID']==10249)&(df_encoded['ProductID']==50842)].sort_values(by = "ActualSaleDate")
chk = df_encoded[(df_encoded['StoreID']==6586)&(df_encoded['ProductID']==45605)].sort_values(by = "ActualSaleDate")
#chk1 = chk.merge(rd,on=['VendorName'],how = 'left')
#chk = chk.drop_duplicates()
columns_to_fill = ['VendorId', 'VendorName', 'RetailerID', 'Retailer']
chk = fill_nans(chk, columns_to_fill)
chk = fill_nans2(chk, columns_to_fill)
#chk['Sales_7_Days_Lag'] = chk['proportion_sale'].shift(7)
#chk['Sales_14_Days_Lag'] = chk['proportion_sale'].shift(14)
chk['Sales_7_Days_Lag'] = chk['QtySold'].shift(7)
chk['Previousday_EOD_Inv'] = chk['Inv_eod'].shift(7)
chk['Previousday_Inv_morn'] = chk['Inv_morn'].shift(7)
cols_selected = ['QtySold','tavg', 'wspd','Sales_7_Days_Lag', 'Day_of_week_Friday', 'Day_of_week_Monday', 'Day_of_week_Saturday','Day_of_week_Sunday', 'Day_of_week_Thursday', 'Day_of_week_Tuesday','Day_of_week_Wednesday',  'Holiday_0', 'Holiday_1','Previousday_EOD_Inv']#,'Inv_morn', ,'Previousday_Inv_morn'
cols_to_drop = list(chk.columns[chk.isna().all()]) # finding columns that have all Nan values
cols_selected = [col for col in cols_selected if col not in cols_to_drop] # removing columns that have all Nan values
chk_train = chk[cols_selected][chk['ActualSaleDate']<=end_date]
chk_test = chk[cols_selected][chk['ActualSaleDate']>end_date]

#Removing outliers and nans from train data
chk_train.dropna(inplace=True)
Q1 = chk_train['QtySold'].quantile(0.25)
Q3 = chk_train['QtySold'].quantile(0.75)
IQR = Q3-Q1
chk_train = chk_train[(chk_train['QtySold'] >= Q1-1.5*IQR) & (chk_train['QtySold'] <= Q3+1.5*IQR)]
chk_train_cts = chk_train['QtySold'].value_counts().reset_index()
label_train = chk_train.pop("QtySold")
label_test = chk_test.pop("QtySold")

# Set a random seed for reproducibility
np.random.seed(475)

#Model fitting
rf = RandomForestRegressor(max_features=0.5)
mdl = rf.fit(chk_train,label_train)

#Prediction on test set
pred = mdl.predict(chk_test)
pred = pd.DataFrame(np.round(pred)).rename(columns={0:'Predicted'})
pred.index = chk_test.index
diff = pred['Predicted'] - label_test


#Prediction on train set
pred_train = mdl.predict(chk_train)
pred_train = pd.DataFrame(np.round(pred_train)).rename(columns={0:'Predicted'})
pred_train.index = chk_train.index
WMAPE_train = np.sum(abs(label_train-pred_train['Predicted'])) / np.sum(label_train)
np.sum(label_train-pred_train['Predicted'])

#Formatting the output and calculating performance metrics
pred.index = chk_test.index
chk.loc[chk.index.isin(pred.index),['Pred']] = pred['Predicted']
chk['Pred'].fillna(chk['QtySold'], inplace=True)
chk.loc[chk.index.isin(pred.index),['Pred_all']] = pred['Predicted']
chk['Pred_all'].fillna(pred_train['Predicted'], inplace=True)
chk['Pred_all'].fillna(chk['QtySold'], inplace=True)# Filling the nans for the 1st 7 days where there is no lag values
#chk['error'] = abs(chk['QtySold']-chk['Pred_all'])
#mae = mean_absolute_error(label_test, pred)
#rmse = np.sqrt(mean_squared_error(label_test, pred['Predicted']))
WMAPE_test = np.sum(abs(label_test-pred['Predicted'])) / np.sum(label_test)

chk[['WMAPE_train','WMAPE_test']] = WMAPE_train,WMAPE_test
one_hot_columns = [col for col in chk.columns if col.startswith(('Day_of_week_', 'Holiday'))]
chk['DAYOFWEEK_NM'] = (chk[[col for col in chk if col.startswith(('Day_of_week_'))]]==1).idxmax(1)
chk['Holiday'] = (chk[[col for col in chk if col.startswith(('Holiday_'))]]==1).idxmax(1)
chk['DAYOFWEEK_NM'] = chk['DAYOFWEEK_NM'].str.replace('Day_of_week_','')
chk['Holiday'] = chk['Holiday'].str.replace('Holiday_','')
chk = chk.drop(columns=one_hot_columns)
(np.sum(chk['Pred']) - np.sum(chk['QtySold']))
np.sum(label_test)
np.sum(pred)

(np.sum(chk['Pred']) - np.sum(chk['QtySold']))/np.sum(chk['Pred'])
np.sum(label_train)+np.sum(label_test)
np.mean(chk['Pred'])

#Adding concsecutive days column
# Create a column to identify consecutive days with zero sales
import copy
chk_bkp = copy.deepcopy(chk)
chk['zero_sales_streak'] = (chk['QtySold'] == 0).astype(int)

# Use the cumsum() function to assign a unique group number to consecutive zero sales streaks
chk['zero_sales_group'] = chk['zero_sales_streak'].cumsum()
# Use the cumsum() function to assign a unique group number to consecutive zero sales streaks
chk['zero_sales_group'] = (chk['QtySold'] != 0).astype(int).cumsum()

# Calculate the count of consecutive zero sales days for each group
chk['consecutive_zero_sales'] = chk.groupby('zero_sales_group')['zero_sales_streak'].cumsum()

# Filter out the rows where sales are not zero
chk['consecutive_zero_sales'] = chk['consecutive_zero_sales'] * (chk['QtySold'] == 0)

# Drop the intermediate columns if needed
chk = chk.drop(['zero_sales_streak', 'zero_sales_group'], axis=1)


########################################################
##FB Prophet############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_squared_error, mean_absolute_error,accuracy_score,r2_score
from tqdm import tqdm_notebook as tqdm
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import warnings
warnings.filterwarnings('ignore')

def fill_nans(df, columns_to_fill):
    for col in columns_to_fill:
        df[col].fillna(method='ffill', inplace=True)
    return df

def fill_nans2(df, columns_to_fill):
    for col in columns_to_fill:
        df[col].fillna(method='bfill', inplace=True)
    return df

df = pd.read_csv("path to the file")
vr_cols = ['VendorId','VendorName','RetailerID','Retailer','StoreID','ProductID','ActualSaleDate']
vr = pd.read_csv("path to vendor info",names=vr_cols)
vr['ActualSaleDate'] = pd.to_datetime(vr['ActualSaleDate'])
ans1 = vr.groupby(['ActualSaleDate','StoreID','ProductID'])['VendorId', 'RetailerID'].nunique().reset_index()
ans1_fin = ans1[ans1['VendorId']<2].drop(columns = ['VendorId', 'RetailerID'])
vr_fin = ans1_fin.merge(vr,on = ['ActualSaleDate', 'StoreID', 'ProductID'],how = 'left')
#chk_vr = vr[(vr['StoreID'] == 152897) & (vr['ProductID'] == 41582)]
df['ActualSaleDate'] = pd.to_datetime(df['ActualSaleDate'])
df = df.merge(vr_fin,on=['StoreID','ProductID','ActualSaleDate'],how = 'left')

df['Day_of_week'] = df['ActualSaleDate'].dt.strftime('%A')
cal = calendar()
holidays = cal.holidays(start='2021-01-01', end='2022-12-31')
df['Holiday'] = df['ActualSaleDate'].isin(holidays)
df['Holiday'] = df['Holiday'].astype(int)

def map_to_season(date):
    if date.month in [3, 4, 5]:
        return 'Spring'
    elif date.month in [6, 7, 8]:
        return 'Summer'
    elif date.month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

# Apply the function to create a 'season' column
df['Season'] = df['ActualSaleDate'].apply(map_to_season)
df_encoded = pd.get_dummies(df, columns=['Holiday', 'Day_of_week','Season'],prefix=['Holiday', 'Day_of_week','Season'], prefix_sep='_')

end_date = pd.to_datetime('2022-06-30')
chk = df_encoded[(df_encoded['StoreID']==71522) &(df_encoded['ProductID']==45605)].sort_values(by = "ActualSaleDate")
columns_to_fill = ['VendorId', 'VendorName', 'RetailerID', 'Retailer']
chk = fill_nans(chk, columns_to_fill)
chk = fill_nans2(chk, columns_to_fill)

chk_train = chk[chk['ActualSaleDate']<=end_date]
chk_test = chk[chk['ActualSaleDate']>end_date]
cols_selected = ['ActualSaleDate','QtySold','Holiday_0','Holiday_1','Day_of_week_Friday','Day_of_week_Monday', 'Day_of_week_Saturday', 'Day_of_week_Sunday','Day_of_week_Thursday', 'Day_of_week_Tuesday', 'Day_of_week_Wednesday','Season_Autumn', 'Season_Spring', 'Season_Summer', 'Season_Winter']#,'proportion_sale'
chk_train = chk_train[cols_selected]
chk_test = chk_test[cols_selected]
chk_train.columns = ['ds','y','Holiday_0','Holiday_1','Day_of_week_Friday','Day_of_week_Monday', 'Day_of_week_Saturday', 'Day_of_week_Sunday','Day_of_week_Thursday', 'Day_of_week_Tuesday', 'Day_of_week_Wednesday','Season_Autumn', 'Season_Spring', 'Season_Summer', 'Season_Winter']#,'proportion_sale'
m = Prophet()
m.fit(chk_train)
future = m.make_future_dataframe(periods=len(chk_test)) #MS for monthly, H for hourly
forecast = m.predict(future)
#plot_plotly(m ,forecast)
pred = pd.DataFrame(forecast.iloc[-len(chk_test):]['yhat'])
pred['yhat']= np.where(pred['yhat'] < 0, 0, pred['yhat'])

#Formatting the output and calculating performance metrics
pred.index = chk_test.index
chk.loc[chk.index.isin(pred.index),['Pred']] = pred['yhat'].round()
chk['Pred'].fillna(chk['QtySold'], inplace=True)
mae = mean_absolute_error(chk_test['QtySold'], pred)
rmse = np.sqrt(mean_squared_error(chk_test['QtySold'], pred['yhat']))
WMAPE = np.sum(abs(chk_test['QtySold']-pred['yhat'])) / np.sum(chk_test['QtySold'])
chk[['RMSE','MAE','WMAPE']] = rmse,mae,WMAPE
one_hot_columns = [col for col in chk.columns if col.startswith(('Day_of_week_', 'Holiday_','Season_'))]
chk['DAYOFWEEK_NM'] = (chk[[col for col in chk if col.startswith(('Day_of_week_'))]]==1).idxmax(1)
chk['Holiday'] = (chk[[col for col in chk if col.startswith(('Holiday_'))]]==1).idxmax(1)
chk['Season'] = (chk[[col for col in chk if col.startswith(('Season_'))]]==1).idxmax(1)
chk['DAYOFWEEK_NM'] = chk['DAYOFWEEK_NM'].str.replace('Day_of_week_','')
chk['Holiday'] = chk['Holiday'].str.replace('Holiday_','')
chk['Season'] = chk['Season'].str.replace('Season_','')
chk = chk.drop(columns=one_hot_columns)
#one_hot_columns = [col for col in chk.columns if col.startswith('Holiday_')]
