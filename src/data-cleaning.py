# Imports
import pandas as pd
import numpy as np
import re

# *** APPLIANCE REPAIR DATA by "Search term" (June 2015 - 2017) 

	'''
	INPUT : 
	filename = appliance-repair-search term-jun-2015-17.csv

	OUTPUT: 
	search-terms-clean.csv
	final_shape: (90178, 12)

	'''
# file load

filepath1 = r"../data/appliance-repair-data/click data/appliance-repair-search term-jun-2015-17.csv"
searches = pd.read_table(filepath1, sep = ',')

# # print statements
# print searches.head()
# print searches.dtypes

#Remove ',' from clicks, impression, and conversions column values 
# and remove '%' from CTR and conv rate and change the data types
clicks = []
impressions = []
CTR = []
avg_cpc = []
cost = []
conversions = []
cpc = []
conv_rate = []

for i in xrange(len(searches)):
	clicks.append(re.sub(',', '', searches.Clicks[i]))
	impressions.append(re.sub(',', '', searches.Impressions[i]))
	CTR.append(searches.CTR[i].strip('%'))
	# Cost
	cost.append(re.sub(',', '', searches.Cost[i]))
	cost[i] = cost[i].strip('$')
	avg_cpc.append(re.sub(',', '', searches['Avg. CPC'][i]))
	avg_cpc[i] = avg_cpc[i].strip('$')
	# Conversions
	conversions.append(re.sub(',', '', searches.Conversions[i]))
	# CPC
	cpc.append(re.sub(',', '', searches['Cost / conv.'][i]))
	cpc[i] = cpc[i].strip('$')
	#Conv. Rate
	conv_rate.append(searches['Conv. rate'][i].strip('%'))

# assigning to the dataframe column
searches.Clicks = clicks
searches.Impressions = impressions
searches.CTR = CTR
searches['Avg. CPC'] = avg_cpc
searches.Cost = cost
searches.Conversions = conversions
searches['Cost / conv.'] = cpc
searches['Conv. rate'] = conv_rate

# change data type to int/float
searches['Clicks'] = searches['Clicks'].astype(int)
searches['Impressions'] = pd.to_numeric(searches['Impressions'])
searches['CTR'] = searches['CTR'].astype(float)
searches['Avg. CPC'] = searches['Avg. CPC'].astype(float)
searches['Cost'] = searches['Cost'].astype(float)
searches['Avg. position'] = searches['Avg. position'].astype(float)
searches['Conversions'] = searches['Conversions'].astype(float)
searches['Conv. rate'] = searches['Conv. rate'].astype(float)
searches['Cost / conv.'] = searches['Cost / conv.'].astype(float)

# # print statements
# print searches.dtypes
# # Data reporting errros ( clicks > impressions) 
# print searches[searches['Clicks'] > searches['Impressions']].Clicks.count()
# print searches[searches['Clicks'] > searches['Impressions']][['Clicks', 'Impressions']].sort_values(by='Clicks').tail(5)


# Lets replace clicks by impressions at erroroneous spots
# dataerrors = searches[searches['Clicks'] > searches['Impressions']][['Clicks', 'Impressions']]
# for i in dataerrors.index:
#     searches['Clicks'].iloc[i] = searches['Impressions'][i]

# # Testing if the values changed correctly
# dataerrors[searches['Clicks'].loc[dataerrors.index] <> searches['Impressions'].loc[dataerrors.index]]

#alternative:
mask = searches['Clicks'] > searches['Impressions']
searches['Clicks'].loc[mask] = searches['Impressions'][mask]

# write to a clean CSV file
searches.to_csv('search-terms-clean.csv', sep = ',', index=False)

# *** USER LOCATIONs - APPLIANCE REPAIR DATA by "User Locations" (June, 1 2015 - June 27, 2017)
	'''
	INPUT: 
	filename = appliance-repair-user locations report.csv
	the first line has file detail text
	second line is the column headers
	each letters are seperated by '^@'

	OUTPUT:
	locations-clean.csv
	final shape: (339365, 13)

	'''

# Delete the first line
#!sed -i '' 1d ../data/appliance-repair-data/click\ data/appliance-repair-user\ locations\ report.csv

# To remove the weird character (^@) that is in between each letters
#!tr < appliance-repair-user\ locations\ report.csv -d '\000' > user_locations_sed.csv

# file upload 
filepath2 = r"../data/appliance-repair-data/click data/user_locations_sed.csv"
locations = pd.read_table(filepath2)
locations.head()

# # print statements
# print locations.shape
# print locations.iloc[339365]
# print locations.dtypes

# Delete the last line
locations = locations[locations['Location'] <> 'Total']

# Change data types:
locations['Conversions'] = locations['Conversions'].astype(float)

# Conv rate
ctr2 = []
# cost2 = []
conv_rate2 = []
for i in xrange(len(locations)):
	ctr2.append(locations['CTR'][i].strip('%'))
	#cost2.append(re.sub(',', '', locations.Cost[i]))
	conv_rate2.append(locations['Conv. rate'][i].strip('%'))
    
locations['CTR'] = ctr2
# locations['Cost'] = cost2
locations['Conv. rate'] = conv_rate2

locations['CTR'] = locations['CTR'].astype(float) 
# locations['Cost'] = locations['Cost'].astype(float)
locations['Conversions'] = locations['Conversions'].astype(float)
locations['Conv. rate'] = locations['Conv. rate'].astype(float)

# print statements:
locations.dtypes

# data errors : clicks > impressions
mask2 = locations['Clicks'] > locations['Impressions']
locations['Clicks'].loc[mask2] = locations['Impressions'][mask2]

# write to a clean CSV file
locations.to_csv('locations-clean.csv', sep = ',', index=False)


# *** CALL DETAILS - APPLIANCE REPAIR DATA by "Call" (June, 1 2015 - June 27, 2017)
	'''
	INPUT:
	Note: 
	filename = appliance-repair-call details report.csv
	the first line has file detail text
	second line is the column headers
	each letters are seperated by '^@'

	OUTPUT:
	call-data-clean.csv
	shape: (278356, 10)

	'''

# Delete the first line of the file in terminal
# sed -i '' 1d ../data/appliance-repair-data/click\ data/appliance-repair-call\ details\ report.csv

# to remove the weird characters in between the letters
# tr < ../data/appliance-repair-data/click\ data/appliance-repair-call\ details\ report.csv -d '\000' > call_details_sed.csv

# File upload
filepath3 = r"call_details_sed.csv"
call_details = pd.read_table(filepath3, header=0)

# # print statements
# print call_details.shape
# print call_details.head()
# print call_details.dtypes

# Data types conversions
# Start time, end time to convert to datetime
call_details['Start time'] = pd.to_datetime(call_details['Start time'])
call_details['End time'] = pd.to_datetime(call_details['End time'])

# write to a clean CSV file
call_details.to_csv('call-details-clean.csv', sep = ',', index=False)

# *** CALL DATA (APPLIANCE REPAIR CALL DATA )  (Sept, 1 2015 - May 31, 2016) & (Sept, 1 2015 - May 31, 2016)
	'''
	INPUT:
	File # 1 : call-report_2015-09-01_to_2016-05-31.csv 
	File # 2 : call-_Report_2016-06-01_to_2017-06-28.xlsx

	shape of file-1: (238459, 30)
	shape of file-2: (137206, 30)

	OUTPUT:
	call-data-clean.csv
	final shape: (375665 , 21)

	'''

# loading files to dataframes
filepath4 = r"../data/appliance-repair-data/call data/call-report_2015-09-01_to_2016-05-31.csv"
data1 = pd.read_table(filepath4, sep = ',')

filepath5 = r"../data/appliance-repair-data/call data/call-_Report_2016-06-01_to_2017-06-28.xlsx"
data2 = pd.read_excel(filepath5, sep = ',')

# # print statements:
# print data1.head()
# print data2.head()
# print data1.shape
# print data2.shape
# print data1.ix[1, :]

# dropping columns consisting of only NAs
data1.dropna(axis=1, how='all', inplace=True)
colnames = data1.columns
data2 = data2[colnames]

# combining both the datasets
data = data1.append(data2, ignore_index=True)

# Datatypes:
data['Call Start Time'] = pd.to_datetime(data['Call Start Time'])
data['Total Duration'] = pd.to_datetime(data['Total Duration'])
data['Connected Duration'] = pd.to_datetime(data['Connected Duration'])
data['IVR Duration'] = pd.to_datetime(data['IVR Duration'])
# write to a clean CSV file
data.to_csv('call-data-clean.csv', index = False)