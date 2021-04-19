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


def get_upcoming_fixtures():
    
    r = requests.get('https://www.theroar.com.au/afl-draw/')
    page = BS(r.content,'html.parser')

    fixtures = []
    for row in page.find_all('tr'):
        try:
            newgamedate = row.find('strong').text.split('-')[0]
            datetime.strptime(newgamedate,'%A, %B %d')
            gamedate = newgamedate
        except:
            pass
            
        if not 'gamedate' in locals(): continue
        
        if not row.find_all('td'): continue
    
        teams = row.find('td').text
        
        if ' vs ' in teams:
            try:
                gametime = row.find_all('td')[2].text
                if gametime == 'TBC': gametime = '3:00pm'
            except:
                gametime = '3:00pm'
            gametime = datetime.strptime(gamedate+' 2021 '+gametime,'%A, %B %d %Y %H:%M%p')
            venue = row.find_all('td')[1].text
            hteam,ateam = teams.split(' vs ')
            if gametime > datetime.now():
                fixtures.append((hteam,ateam,gametime,venue))
                
    return fixtures

def predict_upcoming(k,fixtures):
        
    for match in fixtures:
        
        hteam,ateam = match[:2]
        
        hteam = theroar_to_afl_tables[hteam]
        hteam = theroar_to_afl_tables[ateam]
        print(hteam,ateam)
        
        k.simulate(hteam,ateam)

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
    

if __name__ == '__main__':
##    df = loop()
#    df = pd.read_pickle('df')
#    venues = VenueAPI()
#    df = venues.get_series(df)
#    
#    weather_forecaster = Forecast()

    k = Comp(df.copy(),gamma=0.1)
    
    k.update()
    k.rank_teams()
    
    fixtures = k.df[k.df.index>datetime.now()]

    t = PrettyTable()
    t.field_names = ['','HOME','AWAY']
    
    for ind,match in fixtures.iloc[:9].iterrows():
        home,away = match.NONVARIABLE.HOME, match.NONVARIABLE.AWAY
        
        homevars, awayvars = k.get_active_vars(match)
        
        h,a = [np.round(x,3) for x in k.predict(home,away,homevars,awayvars)]
        hpred = np.array([h]).reshape(1,1)-k.normalize_offset
        realh = k.pt.inverse_transform(hpred)[0]
        apred = np.array([a]).reshape(1,1)-k.normalize_offset
        reala = k.pt.inverse_transform(apred)[0]
        
        hc, ac = [np.round(x*100,2) for x in k.simulate(h,a)]
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
        ["Exp. Score",int(realh),int(reala)],
        ["Chance",f'{hc}%',f'{ac}%'],
        ["Odds",hodds,aodds]
        ])
    
    print(t)
    