import pandas as pd
import datetime
df=pd.read_csv("WHO COVID-19 cases.csv",header=0)
#print(df.columns)
country_code=df['Country_code'].tolist()
#print(country_code)
date_index=df['Date_reported'].tolist()
newd=map(date_index.split(),date_index)
print(list(newd))
#new_d= datetime.datetime(date_inde]).strftime()
#print(new_d)






