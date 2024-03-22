# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:45:38 2023

@author: user
"""

# Importing required packages
import pandas as pd
import numpy as np
from datetime import datetime 
from meteostat import Point, Daily, units  
from meteostat import Stations
import yfinance as yf

def create_weather_data() :
	stations = Stations()
	stations = stations.region('US')
	stations_data = stations.fetch()[['name', 'country', 'region', 'latitude', 'longitude']].reset_index(drop=True)
	#weather_st_year = int(app_settings.external_d_start_date.split('-')[0])
	##weather_end_year = int(app_settings.external_d_end_date.split('-')[0])
	#weather_st_month = int(app_settings.external_d_start_date.split('-')[1])
	#weather_end_month = int(app_settings.external_d_end_date.split('-')[1])
	##weather_st_date = int(app_settings.external_d_start_date.split('-')[2])
	#weather_end_date = int(app_settings.external_d_end_date.split('-')[2])
	chunk_size = 100
	weather_list = []
	for chunk in range(0, stations_data.shape[0], chunk_size) :
		weather_data = pd.DataFrame()
		start_index = chunk
		end_index = chunk + chunk_size
		if end_index > stations_data.shape[0] :
			end_index = stations_data.shape[0]
		print('Iteration chunks :', chunk, start_index, end_index)
		for ind, val in stations_data.iloc[start_index:end_index].iterrows() :
		    print(ind)
		    start = datetime(weather_st_year, weather_st_month, weather_st_date) 
		    end = datetime(weather_end_year, weather_end_month, weather_end_date)  
		    location = Point(val['latitude'], val['longitude'])  
		    data = Daily(location, start, end) 
		    data = data.convert(units.imperial) 
		    data = data.fetch()
		    data = data.reset_index()
		    data['name'] = val['name']
		    data['country'] = val['country']
		    data['region'] = val['region']
		    data['latitude'] = val['latitude']
		    data['longitude'] = val['longitude']
		    
		    weather_data = pd.concat([weather_data, data], axis=0).reset_index(drop=True)
		weather_list.append(weather_data)
	weather_data = pd.concat(weather_list, axis=0)
	temp_data = weather_data[(weather_data['tavg'].notna())].groupby(['country', 'region', 'time'])[['tavg']].quantile(0.5).reset_index()
	precp_data = weather_data[(weather_data['prcp'].notna())].groupby(['country', 'region', 'time'])[['prcp']].quantile(0.5).reset_index()
	weather_data = temp_data.merge(precp_data, on=['country', 'region', 'time'], how='outer').reset_index(drop=True)
	temp_data = pd.pivot_table(weather_data, values = 'tavg', index=['time'], columns = 'region').reset_index()
	temp_data.rename(columns={col:col+"-T" for col in temp_data.columns.values if col not in ['time']}, inplace=True)
	prec_data = pd.pivot_table(weather_data, values = 'prcp', index=['time'], columns = 'region').reset_index()
	prec_data.rename(columns={col:col+"-P" for col in prec_data.columns.values if col not in ['time']}, inplace=True)
	#temp_data.to_csv(input_path + 'HistoricStatesTemperature.csv', index=False)
	#prec_data.to_csv(input_path + 'HistoricStatesPrecipitation.csv', index=False)
	return temp_data, prec_data

create_weather_data()

