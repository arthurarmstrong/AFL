import requests, io, zipfile, re, os, time
from io import StringIO
import pandas as pd, numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup as BS
from weather_au import api
import pickle

headers = {
    'authority': 'scrapeme.live',
    'dnt': '1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site': 'none',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-user': '?1',
    'sec-fetch-dest': 'document',
    'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
}


def find_weather_station_by_coordinates(lat,long):
    url = f'http://www.bom.gov.au/jsp/ncc/cdio/weatherStationDirectory/d?&p_state=&p_display_type=ajaxStnListing&p_nccObsCode=136&p_lat={abs(lat)}&p_lon={abs(long)}&p_stnNum=&p_radius=50&_dc=1618732246941'
    
    resp = requests.get(url,headers=headers)
    
    try:
        df = pd.read_html(resp.text)[0]
    except:
        return
    
    df.index = pd.DatetimeIndex([x for x in df.Last.values])
    
    df = df.loc[[df.index.max()]]

    df = df.loc[df.Km < 10]
    return df.sort_values(by='First').iloc[0].Station

def load_stadium_data():
    with open('StadiumData','rb') as f:
        stadium_data = pickle.load(f)
    return stadium_data

def update_weather_history():
    
    w = Weather()
    
    stadium_data = load_stadium_data()
    
    for stadium,v in stadium_data.items():
        if not v['Station ID']: continue
        
        fname = f'{v["Station ID"]}_rainfall'
        if fname in os.listdir('WeatherData'):
            mod_time = os.path.getmtime(f'WeatherData/{fname}')
            if time.time() - mod_time < 7*24*3600: continue
    
        w.request(v['Station ID'])
        w.save()

class WeatherIndexer:
    
    def __init__(self):
        self.lookup_col = {'temperature':'Maximum temperature (Degree C)',
                           'rainfall':'Rainfall amount (millimetres)'
                           }
    
    def lookup(self,station,lookupdate,obs_type='rainfall'):
        
        if type(lookupdate) == str:
            lookupdate = np.datetime64(lookupdate)
        
        col = self.lookup_col[obs_type]
        
        df = pd.read_pickle(f'WeatherData/{station}_{obs_type}')
        
        return df.asof(lookupdate)[col]
        

class Weather:
    
    def __init__(self,station=None,obs_type='rainfall'):
        
        self.headers = headers
        
        self.obs_codes = {
                          'rainfall':'136',
                          'temperature':'122'
                          }
    
        self.obs_type = self.obs_codes[obs_type]
        
        if station:
            self.station = station
            self.request(station,obs_type)
            self.save()
        
    def request(self,station,obs_type='rainfall'):
        
        self.obs_code = self.obs_codes[obs_type]
        self.obs_type = obs_type
        self.station = station

        url = f'http://www.bom.gov.au/jsp/ncc/cdio/weatherData/av?p_nccObsCode={self.obs_code}&p_display_type=dailyDataFile&p_startYear=&p_c=&p_stn_num={station}'
        resp = requests.get(url,headers=self.headers)
        page = BS(resp.content,'html.parser')
        dl_url = 'http://www.bom.gov.au'+page.find('a',text='All years of data').get('href')
        
        resp = requests.get(dl_url,headers=self.headers)
        
        bio = io.BytesIO(resp.content)
        zf = zipfile.ZipFile(bio)
        CSV = zf.read(zf.filelist[0])
        weatherstring = StringIO(CSV.decode())
        weather = pd.read_csv(weatherstring)
        
        weather.set_index(pd.DatetimeIndex([f'{x.Month}/{x.Day}/{x.Year}' for row,x in weather.iterrows()]),inplace=True)
        
        self.observations = weather
    
    def save(self):
        
        self.observations.to_pickle(f'WeatherData/{self.station}_{self.obs_type}')
                
class Forecast:
    
    def __init__(self):
        pass
    
    def get(self,loc):
        
        self.API = api.WeatherApi(search=loc.replace(' ','+'), debug=0)
        
        self.daily_forecast = self.API.forecasts_daily()
        self.forecast_3hr = self.API.forecasts_3hourly() 
        
    def date_forecast(self,day,forecast_type='daily'):
        day = datetime.strptime(day,'%Y-%m-%d %H:%M')
        if forecast_type == 'daily':
            return sorted(self.daily_forecast,key=lambda x: abs(day-datetime.strptime(x['date'],'%Y-%m-%dT%H:%M:%SZ')))[0]
        elif forecast_type == '3hr':
            return sorted(self.forecast_3hr,key=lambda x: abs(day-datetime.strptime(x['time'],'%Y-%m-%dT%H:%M:%SZ')))[0]
        
if __name__ == '__main__':

    
    update_weather_history()
    i = WeatherIndexer()
    
        