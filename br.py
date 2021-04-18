from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
import re,os
from bs4 import BeautifulSoup as BS
import pandas as pd
import sqlite3, dateparser
import numpy as np
import time
from datetime import datetime
from consolidate_data import consolidate_data

def get_br_results(browser=None,waitForSelections=False,sport='Aussie rules',catsel='Australia',event='All tournaments'):
        
    def pull_results(existing_games,browser,weeks=2):
        ref_timestamp = time.time()
        
        for wk in range(weeks):
            #Set date range  
            today_time_struct = time.gmtime(ref_timestamp)
            from_date_struct = time.gmtime(ref_timestamp - 6*24*3600)

            from_date = time.strftime("%Y-%m-%d",from_date_struct)
            to_date = time.strftime("%Y-%m-%d",today_time_struct)
            browser.execute_script('document.getElementById("fromDate").value = "'+from_date+'"')
            browser.execute_script('document.getElementById("toDate").value = "'+to_date+'"')
            
            #click search button
            browser.find_element_by_name('go').click()
            time.sleep(2)
            
            page = BS(browser.page_source,'html.parser')
            existing_games = get_data_from_table(existing_games,page)
                    
            ref_timestamp -= 7*24*3600
            
        return existing_games
    
    
    while True:
        try:
            #once logged in, click results tab
            browser.find_element_by_partial_link_text('Resulting').click()
            browser.find_element_by_link_text('Results').click()
            break
        except:
            pass
    
    try:
        browser.switch_to_frame('innerframe')
    except:
        pass
    
    while True:
        try:
            browser.find_element_by_link_text('Advanced version').click()
            break
        except:
            pass

    while True:
        try:
            if waitForSelections:
                input('Please make your selections and press enter.')
            else:
                #Else automatically select the 
                sportsel_select = browser.find_element_by_id('sportsel_adv')
                catsel_select = browser.find_element_by_id('catsel_adv')
                days_select = browser.find_element_by_id('days')
                event_select = browser.find_element_by_id('toursel_adv')
                
                #Click the entire last week button
                [x for x in days_select.find_elements_by_tag_name('option')][-1].click()
                #Click into sport
                [x for x in sportsel_select.find_elements_by_tag_name('option') if x.get_attribute('text') == sport][0].click()
                #time.sleep(2)
                #click into Electronic League
                [x for x in catsel_select.find_elements_by_tag_name('option') if x.get_attribute('text') == catsel][0].click()
                #time.sleep(2)
                #click into requested event
                time.sleep(1)
                existing_games = opendf(f'{catsel}{event}')
                matching_events = []
                for x in event_select.find_elements_by_tag_name('option'):
                    this_event_name = x.get_attribute('text')
                    if event in this_event_name:
                        matching_events.append(x)
                for ev in matching_events:
                    ev.click()
                    existing_games = pull_results(existing_games,browser)
                #time.sleep(2)
                
            break
        except:
            pass
            
    existing_games = fix_names(existing_games)
    existing_games = consolidate_data(existing_games)
    
    existing_games.drop_duplicates(subset=['GAME ID'],inplace=True,keep='last')
    
    
    conn = connect_to_database(f'{catsel}{event}.db')
    existing_games.to_pickle(f'{catsel}{event}')
    existing_games.to_sql('MATCHES',conn,index=False)
    conn.commit()
    conn.close()
    
    
def get_data_from_table(existing_games,page):
    
    tables = page.find_all('table',{'class':'searchResultTable'})
    if not tables: return existing_games

    for table in tables:
        
        datestr = table.find('tbody').find('td').text.strip()
        
        
        for row in table.find_all('tr')[1:]:
                        
            cols = row.find_all('td')
            
            try:
                if 'font-weight:bold' in cols[0].get('style'):
                    compname = cols[0].text.replace(u'\xa0', ' ').replace('  ',' ').strip(' -')
                    continue
            except:
                pass

            try:
                gmtime = cols[1].text
            except:
                continue
            
            gametimestr = datestr+' '+gmtime
            utctime_struct = datetime.utctimetuple(dateparser.parse(gametimestr))
            time_stamp = time.mktime(utctime_struct)
            

            hometeam,awayteam,fix_names = get_names(cols[0].text)
            homescore,awayscore = [int(x) for x in cols[2].text.split(':')]
            gameid = str(time_stamp)+hometeam+awayteam


            gamedic= {
            'DATE':[datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M')],
            'TIMESTAMP':[time_stamp],
            'COMPETITION':[compname],
            'HOME':[hometeam],
            'AWAY':[awayteam],
            'VENUE':[np.nan],
            'HOME SCORE':[homescore],
            'AWAY SCORE':[awayscore],
            'TOTAL':[homescore+awayscore],
            'SEASON':[utctime_struct[0]],
            'GAME ID':[gameid],
            'FIX NAMES':[fix_names],
            'RAW NAMES STRING':[cols[0].text],
            'STAGE':['Group Stage'],
                       }
                                                                 
            #This turns the new game data into a pandas dataframe
            new_game = pd.DataFrame(gamedic)
            #This adds the new dataframe to the complete list of games
            existing_games = pd.concat([existing_games,new_game],ignore_index=True,sort=False)
    return existing_games 

def opendf(df):
    if df in os.listdir():
        return pd.read_pickle(df)
    else:
        return pd.DataFrame([])
    
def get_names(s):
    #Team names may not be able to be parsed. In that case save a note in boolean form that will indicate it needs postprocessing
    try:
        hometeam,awayteam = [x.strip() for x in s.split(' - ')]
        
        bracket_name = re.findall('\(.*\)',hometeam)
        if bracket_name:
            hometeam = bracket_name[0].strip('() -')
            
        bracket_name = re.findall('\(.*\)',awayteam)
        if bracket_name:
            awayteam = bracket_name[0].strip('() -')
        
        
        return hometeam,awayteam,False
    except:
        return 'Unknown','Unknown',True

def fix_names(df):
    unique_names = set(df['HOME'].values).union(set(df['AWAY'].values))
    
    for i in df.index.values:
        thisgame = df.loc[i]
        
        if thisgame['FIX NAMES']:
            name_matches = []
            for n in unique_names:
                if n in thisgame['RAW NAMES STRING']:
                    name_matches.append(n)
        
            if len(name_matches) == 2:
                m1_pos = thisgame['RAW NAMES STRING'].find(name_matches[0])
                m2_pos = thisgame['RAW NAMES STRING'].find(name_matches[1])
                if m1_pos < m2_pos:
                    df.at[i,'HOME'] = name_matches[0]
                    df.at[i,'AWAY'] = name_matches[1]
                elif m1_pos > m2_pos:
                    df.at[i,'HOME'] = name_matches[1]
                    df.at[i,'AWAY'] = name_matches[0]
                df.at[i,'GAME ID'] = str(thisgame['TIMESTAMP'])+df.at[i,'HOME']+df.at[i,'AWAY']
                df.at[i,'FIX NAMES'] = False
                
    return df    
    
def connect_to_database(path):
    try:
        open(path,'w').close()
    except:
        pass

    conn = sqlite3.connect(path)
    
    return conn

def set_up_browser():
    
    browser = webdriver.Chrome(executable_path='C:\\Users\\GerardArmstrong\\Documents\\Python Scripts\\Compiler\\chromedriver.exe')
    
    browser.get('https://in.betradar.com/betradar/index.php')
    
    #click login button
    #browser.find_element_by_link_text('Login').click()
    
    browser.execute_script('document.getElementById("username").value = "radar491"; document.getElementById("password").value = "Jjr5apbDrRHAT.x"')
    #click submit
    [x for x in browser.find_elements_by_tag_name('button') if x.get_attribute('type') == 'submit'][0].click()

    return browser

if __name__ == '__main__':
    
    browser = set_up_browser()
    
    get_br_results(browser,event='eSports Battle')
    
    #browser.get('https://in.betradar.com/betradar/index.php')
    
    #get_br_results(browser,event='eSports Battle')
    
    browser.close()