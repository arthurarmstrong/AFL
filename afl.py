from bs4 import BeautifulSoup as BS
import requests, pandas as pd, numpy as np, re, pickle
from datetime import datetime
from fancy_baseliner import Comp
from prettytable import PrettyTable
from sklearn.preprocessing import PowerTransformer, StandardScaler
from bom import Weather, WeatherIndexer, Forecast
from venues import VenueAPI

def loop():
    
    df = []
    
    thisyear = datetime.now().year
    
    for season in range(thisyear-10,thisyear+1):
        
        page = get(f'https://www.footywire.com/afl/footy/ft_match_list?year={season}')
    
        for rows in page.find_all('tr'):
            try:
                cols = rows.find_all('td')
                gamedate = datetime.strptime(rows.find('td').text.strip()+ f' {season}','%a %d %b %H:%M%p %Y')
                
                home,away = [a.text.strip() for a in rows.find_all('a')][:2]
                venue = cols[2].text.strip()
                try:
                    h_s,a_s = [int(x) for x in cols[4].text.split('-')]
                except:
                    h_s = a_s = np.nan
                
                if gamedate.year == 2020:
                    h_s /= 0.8
                    a_s /= 0.8
                
                row = {
                        'DATE':gamedate,
                        'HOME':home,
                        'HOME SCORE':h_s,
                        'AWAY':away,
                        'AWAY SCORE':a_s,
                        'VENUE':venue,
                        'TOTAL':h_s+a_s
                        }
                        
                df.append(row)
                
            except:
                continue
            
    df = pd.DataFrame(df).set_index('DATE')
    df.columns = pd.MultiIndex.from_product([['NONVARIABLE'],df.columns])
    return df
    
def get(url):
    
    page = requests.get(url).content
    page = BS(page,'html.parser')
    
    return page

def load_stadium_data():
    with open('StadiumData','rb') as f:
        stadium_data = pickle.load(f)
    return stadium_data

def add_rainfall(df):
    stadium_data  = load_stadium_data()
    
    weather = WeatherIndexer()
    
    for i,gamedate in enumerate(df.index.values):
        
        data = df.iloc[i]
        
        venue = data['VENUE']
        
        sid = stadium_data[venue]['Station ID']
        
        if not sid: continue
        
        rainfall = weather.lookup(sid,gamedate)
        
        data['Rain'] = rainfall
        df.iloc[i] = data
    
    return df

def run(download=False):
      
    df = pd.read_pickle('df')
    
    if download or not df[(df.index<datetime.now()) & df.NONVARIABLE.TOTAL.isnull()].empty:
        df = loop()
    
    venues = VenueAPI()
    df = venues.get_series(df)

    k = Comp(df.copy(),gamma=0.1)
    
    k.update()
    k.rank_teams()

    t = PrettyTable()
    t.field_names = ['','HOME','AWAY']
    
    for ind,match in k.upcoming_fixtures.iloc[:9].iterrows():
        home,away = match.NONVARIABLE.HOME, match.NONVARIABLE.AWAY
                
        hc, ac, h_exp, a_exp = k.simulate(match)
        hodds, aodds = [f'${np.round(100/x,2)}' for x in [hc,ac]]
        
#        venue_address = venues.stadium_data[match.NONVARIABLE.VENUE]['Location'].address.split(',')[2].strip().replace(' ','+')
#        try:
#            forecast = weather_forecaster.get(loc=venue_address)
#            if forecast:
#                print(forecast.date_forecast(ind))
#        except:
#            continue
        

        t.add_rows([
        ['',home,away],
        ["Exp. Score",int(h_exp),int(a_exp)],
        ["Chance",f'{hc}%',f'{ac}%'],
        ["Odds",hodds,aodds]
        ])
    
    print(t)
    
    return k
    

if __name__ == '__main__':
    
    k = run()
    