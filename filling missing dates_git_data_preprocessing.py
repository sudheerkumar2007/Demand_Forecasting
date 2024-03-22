# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:05:36 2023

@author: user
"""
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm_notebook as tqdm
import warnings
warnings.filterwarnings('ignore')

cols = ['StoreNumber','storeCity','storeState','StoreZip5','ProductID','str_sku_id','ProductName','RetailPrice','ActualSaleDate','VendorId','VendorName','RetailerID','Retailer','ProductKey','StoreKey','FormattedDate','QOHCalculated','QtyDelivered','QtyReturned','QtySold']
#cols = ['StoreID','storeCity','storeState','StoreZip5','ProductID','str_sku_id','ProductName','RetailPrice','ActualSaleDate','State_Sales','StartDate','StopDate','ProductKey','StoreKey','FormattedDate','QOHCalculated','QtyDelivered','QtyReturned','QtySold']
orig_df = pd.read_csv("Path to file",names=cols)
orig_df['Store_ID'] = np.where(orig_df['StoreKey'].notna(), orig_df['StoreKey'], orig_df['StoreID'])
orig_df = orig_df.drop(columns = ['StoreKey','StoreID']).rename(columns = {'Store_ID':'StoreID'})
orig_df['Product_ID'] = np.where(orig_df['ProductKey'].notna(), orig_df['ProductKey'], orig_df['ProductID'])
orig_df = orig_df.drop(columns = ['ProductKey','ProductID']).rename(columns = {'Product_ID':'ProductID'})
df = orig_df.drop_duplicates() # No difference in the counts is observed
ans1 = df.groupby(['ActualSaleDate','StoreID','ProductID'])['VendorID', 'RetailerID'].nunique().reset_index()
ans1_fin = ans1[ans1['VendorID']<2].drop(columns = ['VendorID', 'RetailerID'])
df = df.merge(ans1_fin,on=['StoreID','ProductID','ActualSaleDate'],how = 'left')
df['ActualSaleDate'] = pd.to_datetime(df['ActualSaleDate'])
df[['StoreID','ProductID']] = df[['StoreID','ProductID']].astype('int').astype('string')
df[['QOHCalculated', 'QtyDelivered', 'QtyReturned', 'QtySold']] = df[['QOHCalculated', 'QtyDelivered', 'QtyReturned', 'QtySold']].fillna(0)
df['Sale_Date'] = np.where(df['FormattedDate'].notna(), df['FormattedDate'], df['ActualSaleDate'])
df = df.drop(columns = ['FormattedDate','ActualSaleDate']).rename(columns = {'Sale_Date':'ActualSaleDate','QOHCalculated':'Inv_eod'})
max_sale_date = df.groupby(['StoreID','ProductID'])['ActualSaleDate'].max().reset_index().rename(columns = {"ActualSaleDate":"Max_ActualSaleDate"})
df = df.merge(max_sale_date,on = ['StoreID','ProductID'],how = 'left')
df['Max_ActualSaleDate'] = pd.to_datetime(df['Max_ActualSaleDate'])
df = df[df['Max_ActualSaleDate']>pd.to_datetime('2022-06-30')].drop(columns = ['Max_ActualSaleDate']) #Removing data which has no sales in Q3
df['QtySold'] = np.where((df['QtySold'] <0),0,df['QtySold'])
df['Inv_eod'] = np.where((df['Inv_eod'] <0),0,df['Inv_eod'])
df['str_sku_id'] = df['StoreID'] + '-' + df['ProductID']

Zip_sales = pd.DataFrame(df.groupby(['StoreZip5','ProductID'])['QtySold'].sum()).reset_index().rename(columns = {'QtySold':'unitssolds_in_zip'})
df1 = df.merge(Zip_sales,on = ['StoreZip5','ProductID'],how = 'left')
ct_chk = df1.groupby('str_sku_id')['ActualSaleDate'].count()
chk = df1[(df1['StoreID'] == '12427') & (df1['ProductID'] == '45606')]
df_pp = copy.deepcopy(df)
df = chk


#chk = df[(df['StoreID']==10004) &(df['ProductID']==45616)].sort_values(by = "ActualSaleDate")

def load_shipment_data(sale_data) :


	# Filling intermittent rows
    sale_data['ProductID'] = sale_data['ProductID'].astype(str)
    sale_data['StoreID'] = sale_data['StoreID'].astype(str)
    sale_data['ActualSaleDate'] = pd.to_datetime(sale_data['ActualSaleDate'])
    min_order_data = sale_data.groupby(['StoreID','ProductID'])[['ActualSaleDate']].min().reset_index()
    max_order_data = sale_data.groupby(['StoreID','ProductID'])[['ActualSaleDate']].max().reset_index()
    min_order_data.rename(columns={'ActualSaleDate':'MinOrderDate'}, inplace=True)
    max_order_data.rename(columns={'ActualSaleDate':'MaxOrderDate'}, inplace=True)

    max_order_data = max_order_data.merge(min_order_data, on=['StoreID','ProductID'])

    alt_sale_data = pd.DataFrame()
    for ind, val in max_order_data.iterrows() :
        date_list = pd.date_range(val['MinOrderDate'], val['MaxOrderDate'])

        temp_df = pd.DataFrame()
        temp_df['OrderDateRaw'] = date_list
        temp_df['ActualSaleDate'] = pd.to_datetime(temp_df['OrderDateRaw']).dt.normalize()
        temp_df['ProductID'] = val['ProductID']
        temp_df['StoreID'] = val['StoreID']
        temp_df['OrderDate'] = pd.to_datetime(temp_df['ActualSaleDate'])

        temp_df = temp_df[['StoreID','ProductID', 'ActualSaleDate']].drop_duplicates().reset_index(drop=True)
        temp_df = temp_df.merge(sale_data[(sale_data['ProductID'] == val['ProductID']) & (sale_data['StoreID'] == val['StoreID'])], on=['StoreID','ProductID', 'ActualSaleDate'], how='left')

        alt_sale_data = pd.concat([alt_sale_data, temp_df], axis=0)

    alt_sale_data[['QtySold', 'QtyDelivered', 'QtyReturned']] = alt_sale_data[['QtySold', 'QtyDelivered', 'QtyReturned']].fillna(0)

    return alt_sale_data

def treat_outliers(col,df):
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = df['col'].quantile(0.25)
    Q3 = df['col'].quantile(0.75)

    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Impute outlier values with 0
    df['col'] = df['col'].apply(lambda x: 0 if x < lower_bound or x > upper_bound else x)


def fill_nans(df, columns_to_fill):
    for col in columns_to_fill:
        df[col].fillna(method='ffill', inplace=True)
    return df

def fill_nans2(df, columns_to_fill):
    for col in columns_to_fill:
        df[col].fillna(method='bfill', inplace=True)
    return df

def preprocessing(chk):
    chk = load_shipment_data(chk)
    columns_to_fill = ['storeCity', 'storeState', 'StoreZip5', 'str_sku_id', 'ProductName', 'RetailPrice','Inv_eod','VendorID','VendorName','RetailerID','RetailerName','unitssolds_in_zip']
    # Fill NaN values in the specified columns
    chk = fill_nans(chk, columns_to_fill)
    chk = fill_nans2(chk, columns_to_fill)
    # Calculate the Inventory
    chk['Inv_morn'] = chk['Inv_eod'].shift(1, fill_value=0) + chk['QtyDelivered'] - chk['QtyReturned']
    #dummy_start_date = pd.to_datetime('2021-05-31')
    #df = df[df['ActualSaleDate']>dummy_start_date]
    #chk['Inv_eod'] = np.where((chk['Inv_eod']>chk['Inventory']),chk['Inventory'] - chk['QtySold'],chk['QOHCalculated'])
    return chk

grouped_data = df1.groupby('str_sku_id')
# Create a list of DataFrames using dictionary comprehension
data_groups = [group_df.copy() for _, group_df in grouped_data]

product_list = list(df['str_sku_id'].unique())
nan_in_each_column = orig_df.isna().any()

df_fin = []
for product in tqdm(product_list[0:100]):
    
    samp_data = df1[df1['str_sku_id'] == product]

    try:
        samp_data = preprocessing(samp_data)
        df_fin.append(samp_data)

    except:
        print(product)
        continue
    
df_fin = pd.concat(df_fin)
df_fin['chk'] = np.where((df_fin['QtySold']<=df_fin['Inv_morn']),"Yes","No")
PK = df_fin['str_sku_id'][df_fin['chk']=="No"].drop_duplicates()
filled_data = load_shipment_data(chk)
df = filled_data
#, 'unitssolds_in_zip', 'Inv_morn'
columns_to_fill = ['storeCity', 'storeState', 'StoreZip5', 'str_sku_id', 'ProductName', 'RetailPrice','Inv_eod','VendorID','VendorName','RetailerID','RetailerName','unitssolds_in_zip']

# Fill NaN values in the specified columns
alt_sale_data2 = fill_nans(filled_data, columns_to_fill)
alt_sale_data2 = fill_nans2(filled_data, columns_to_fill)

#zip_sale = df[['ActualSaleDate','ProductID','StoreZip5','unitssolds_in_zip']].drop_duplicates()
#alt_sale_data2 = alt_sale_data2.merge(zip_sale,on = ['ActualSaleDate','ProductID','StoreZip5'],how = 'left')
#alt_sale_data2['unitssolds_in_zip'] = np.where(alt_sale_data2['unitssolds_in_zip_x'].notna(), alt_sale_data2['unitssolds_in_zip_x'], alt_sale_data2['unitssolds_in_zip_y'])
#alt_sale_data2 = alt_sale_data2.drop(columns = ['unitssolds_in_zip_x','unitssolds_in_zip_y'])
#alt_sale_data2['unitssolds_in_zip'] = alt_sale_data2['unitssolds_in_zip'].fillna(0)
#alt_sale_data2 = alt_sale_data2.rename(columns = {'QOHCalculated':'Inv_eod'})#,'Inventory':'Inv_morn'

# Calculate the Inventory
alt_sale_data2['Inventory_Morn'] = alt_sale_data2['Inv_eod'].shift(1, fill_value=0) + alt_sale_data2['QtyDelivered'] - alt_sale_data2['QtyReturned']
#df['Inventory'] = np.where((df['Inventory'] <0),(df['QOHCalculated'].shift(1, fill_value=0)+df['QtyDelivered']),df['Inventory'])
df['Inventory'] = np.where((df['Inventory_Morn'] <df['QtySold']),df['QtySold'],df['Inventory_Morn'])
df['unitssolds_in_zip'] = np.where((df['unitssolds_in_zip'] <0),0,df['unitssolds_in_zip'])
dummy_start_date = pd.to_datetime('2021-05-31')
df = df[df['ActualSaleDate']>dummy_start_date]
df['QOHCalculated'] = np.where((df['QOHCalculated']>df['Inventory']),df['Inventory'] - df['QtySold'],df['QOHCalculated'])

#filled_data1 = copy.deepcopy(filled_data)

# Columns with NaN values that you want to fill
alt_sale_data2['f_chk'] = alt_sale_data2['Inv_eod']<=alt_sale_data2['Inv_morn']
alt_sale_data2['Inv_eod'] = np.where((alt_sale_data2['Inv_eod']>alt_sale_data2['Inv_morn']),alt_sale_data2['Inv_morn'] - alt_sale_data2['QtySold'],alt_sale_data2['Inv_eod'])
alt_sale_data2 = fill_nans2(alt_sale_data2, columns_to_fill)
alt_sale_data2 = alt_sale_data2.drop_duplicates()
alt_sale_data2 = alt_sale_data2[['ActualSaleDate','StoreID', 'ProductID', 'str_sku_id',  'ProductName', 'RetailPrice', 'storeCity', 'storeState','StoreZip5', 'Inv_morn','Inv_eod', 'QtySold', 'unitssolds_in_zip']]
#alt_sale_data2 = alt_sale_data2.merge(max_sale_date,on = ['StoreID','ProductID'],how = 'left')
#alt_sale_data3 = alt_sale_data2[alt_sale_data2['Max_ActualSaleDate']>pd.to_datetime('2022-06-30')] # Removing data where there is no sale in Q3
alt_sale_data4 = copy.deepcopy(alt_sale_data3)

###########Adding Weather info###########
import pandas as pd
import numpy as np
weather_data = pd.read_csv("Path to weather info")
weather_data['name'] = weather_data['name'].apply(lambda x: x.lower())
alt_sale_data3 = pd.read_csv("path to file")
cities = pd.DataFrame(np.unique(alt_sale_data3['storeCity'])).rename(columns = {0:'city_name'})
cities['city_name'] = cities['city_name'].apply(lambda x: x.lower())
weather_cities = pd.DataFrame(weather_data['name'].drop_duplicates())

city_name_mapping = {}
for city in cities['city_name']:
    matching_cities = [weather_city for weather_city in weather_cities['name'] if city in weather_city]

    if matching_cities:
        city_name_mapping[city] = matching_cities

# Convert the dictionary to a list of dictionaries
data = [{'Key': key, 'Value': value} for key, values in city_name_mapping.items() for value in values]

# Create a DataFrame from the list of dictionaries
df_city_map = pd.DataFrame(data).rename(columns = {'Key':'storeCity','Value':'name'})

weather_data_new = weather_data.merge(df_city_map,on=['name'],how = 'left')
weather_data_new = weather_data_new.dropna(subset=['storeCity'])
weather_data_new = weather_data_new.sort_values(by = ['storeCity','time'])
weather_data_fin = weather_data_new.groupby(['time','storeCity'])['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt','pres', 'tsun'].mean().reset_index()
weather_data_fin['time'] = pd.to_datetime(weather_data_fin['time'])
weather_data_fin = weather_data_fin.rename(columns = {'time':'ActualSaleDate'})
weather_data_fin['snow'] =  weather_data_fin['snow'].fillna(0)
alt_sale_data5 = alt_sale_data3.merge(weather_data_fin[['ActualSaleDate', 'storeCity', 'tavg','prcp','wspd','snow']],on = ['ActualSaleDate', 'storeCity'],how = 'left' )
alt_sale_data5['unitssolds_in_zip'] = np.where(alt_sale_data5['unitssolds_in_zip']<alt_sale_data5['QtySold'],alt_sale_data5['QtySold'],alt_sale_data5['unitssolds_in_zip'])
alt_sale_data5['proportion_sale'] = alt_sale_data5['QtySold']/alt_sale_data5['unitssolds_in_zip']
alt_sale_data5['proportion_sale'] = np.where(alt_sale_data5['proportion_sale'] == np.Inf, 1, alt_sale_data5['proportion_sale'])
alt_sale_data5.to_csv("C:/Users/user/Downloads/Nexxus related/Nexxus_Jun2021 to Sept2022_full_data_20 stores.csv",index=False)












