import pandas as pd
import re
from fuzzywuzzy import fuzz
import winsound

def str_similarity(r,s,p=None,common_words=None,method='fuzzywuzzy'):
        
    #r is the string being tested
    r = re.sub('[\W_]+', ' ',r)
    s = re.sub('[\W_]+', ' ',s)
    r = r.upper()
    s = s.upper()
    if common_words:
        for w in common_words:
            r = r.replace(w,'')
            s = s.replace(w,'')
    r = r.strip()
    s = s.strip()
    
    if method == 'fuzzywuzzy':
        if not p: p = 85
        matchratio  = fuzz.token_set_ratio(r,s)

    elif method == 'armstrong':
        if not p: p = 0.32
        c = 0
        t = 0
    
        #start at the length of the string and go back to length of 3
        #i is the length of substring
        for i in range(len(r),2,-1):
            #j is the ordinal position
            for j in range(len(r)-i+1):
                if r[j:j+i] in s: c += 1
                t += 1
            
        if t > 0:
            matchratio = float(c)/t
        else:
            matchratio = 0
            
        c = 0
        t = 0             
        for i in range(len(s),2,-1):
            #j is the ordinal position
            for j in range(len(s)-i+1):
                if s[j:j+i] in r: c += 1
                t += 1
                
        #Average the two results
        if t > 0:
            matchratio = (float(c)/t+matchratio)/2
        else:
            matchratio = 0
        
    if matchratio > p:
        matchratio = True
    else:
        matchratio = False
        
    return matchratio

    

def consolidate_data(all_games,p=0.32,method='fuzzywuzzy'):
    
    def is_in_dna(df,t):
        #This function takes a tuple and checks whether its values are in the do not ask dataframe
        for i in df.values:
            if (i[0] == t[0] and i[1] == t[1]) or (i[1] == t[0] and i[0] == t[1]):
                return True   
        return False
    
    def get_common_words():
        words = {}
        for w in unique_teams:
            s = w.split(' ')
            for a in s:
                if not a in list(words.keys()):
                    words[a] = float(1)
                else:
                    words[a] += 1
        for k in list(words.keys()):
            words[k] /= len(words)
            words[k] = round(words[k],4)
        
        return words
        
    print('Consolidating data. Looking for teams with similar sounding names...')
    
    donotask = pd.read_csv('donotask.csv',header=None)
    
    unique_teams = set(all_games['HOME'].values).union(set(all_games['AWAY'].values))

    common_words = get_common_words()
    common_words = [x for x,v in list(common_words.items()) if v > 0.1]
    if common_words: print('Common words that will be ignored:',', '.join([x for x in common_words]))
    
    remaining = set(all_games['HOME'].values).union(set(all_games['AWAY'].values))
    chosen_names = set()
    
    #Create a record of whether the console has beeped to make sure it only does it once
    already_beeped = False
    
    while remaining:
        t = remaining.pop()
        similarities = [(x,t) for x in remaining if str_similarity(t,x,common_words=common_words)]
        if similarities:
            for i in similarities:
                
                if i[0] not in unique_teams or i[1] not in unique_teams: continue
            
                print(f'changing {i[1]} to {i[0]}')
                all_games['HOME'].replace(i[1],i[0],inplace=True)
                all_games['AWAY'].replace(i[1],i[0],inplace=True)

    
    #donotask.to_csv('donotask.csv',header=False,index=False)
    return all_games