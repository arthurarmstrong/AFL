from bs4 import BeautifulSoup as BS
import requests, pandas as pd, numpy as np, re, pickle
from datetime import datetime
from fancy_baseliner import Comp
from prettytable import PrettyTable
from sklearn.preprocessing import PowerTransformer, StandardScaler
from bom import Weather, WeatherIndexer

def loop():
    
    df = []
    
    page = get('https://afltables.com/afl/seas/season_idx.html')
    
    seasonlinks = ['https://afltables.com/afl/seas/'+a.get('href') for a in page.find('table').find_all('a')]
    
    for season in seasonlinks[-10:]:
        
        page = get(season)
    
        for t in page.find_all('table'):
            try:
                rows = t.find_all('tr')
                ht,at = rows[:2]
                
                if not 'Match stats' in at.text: continue
                
                info = ht.find_all('td')[3].text
                dat = re.findall('.*?[AP]M',info)[0]
                dat = datetime.strptime(dat,'%a %d-%b-%Y %H:%M %p')
                                
                venue = re.findall('(?<=Venue: ).*',info)[0]
        
                home = ht.find('td').text
                away = at.find('td').text
                h_s = int(ht.find_all('td')[2].text)
                a_s = int(at.find_all('td')[2].text)
                
                if dat.year == 2020:
                    h_s /= 0.8
                    a_s /= 0.8
                
                row = {
                        'DATE':dat,
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
            
    return pd.DataFrame(df).set_index('DATE')
    
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

def add_travel(df):
    stadium_data  = load_stadium_data()
    
#    homelat, homelong = 
    

if __name__ == '__main__':
    
    newdf = add_rainfall(df)
    
    
#    theroar_to_afl_tables = {'Adelaide Crows': 'Adelaide',
# 'Brisbane Lions': 'Brisbane Lions',
# 'Carlton': 'Carlton',
# 'Collingwood': 'Collingwood',
# 'Essendon': 'Essendon',
# 'Fremantle': 'Fremantle',
# 'GWS Giants': 'Greater Western Sydney',
# 'Geelong Cats': 'Geelong',
# 'Gold Coast Suns': 'Gold Coast',
# 'Hawthorn': 'Hawthorn',
# 'Melbourne': 'Melbourne',
# 'North Melbourne': 'North Melbourne',
# 'Port Adelaide': 'Port Adelaide',
# 'Richmond': 'Richmond',
# 'St Kilda': 'St Kilda',
# 'Sydney Swans': 'Sydney',
# 'West Coast Eagles': 'West Coast',
# 'Western Bulldogs': 'Western Bulldogs'}
#    
#    df = loop()
#    fixtures = get_upcoming_fixtures()
#    
#    k = Comp(df,gamma=0.1)
#    
#    #k.df = df.copy()
#    k.df.loc[:,['HOME SCORE','AWAY SCORE']] = k.pt.fit_transform(k.df.loc[:,['HOME SCORE','AWAY SCORE']])+5
#    k.df['TOTAL'] = k.df['HOME SCORE'] + k.df['AWAY SCORE']
#    
#    k.update()
#    k.rank_teams()
#    
#    
#    t = PrettyTable()
#    t.field_names = ['','HOME','AWAY']
#    
#    for match in fixtures[:9]:
#        home,away = match[:2]
#        
#        home = theroar_to_afl_tables[home]
#        away = theroar_to_afl_tables[away]
#        
#        h,a = [np.round(x,3) for x in k.predict(home,away)]
#        pred = np.array([[h,a]]).reshape(1,2)-5
#        h,a = k.pt.inverse_transform(pred)[0]
#        
#        hc, ac = [np.round(x*100,2) for x in k.simulate(h,a)]
#        hodds, aodds = [f'${np.round(100/x,2)}' for x in [hc,ac]]
#        
#
#        t.add_rows([['',home,away],
#        ["Exp. Score",int(h),int(a)],
#        ["Chance",f'{hc}%',f'{ac}%'],
#        ["Odds",hodds,aodds]])
#    
#    print(t)
    