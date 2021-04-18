
import os, pandas as pd, numpy as np
from sklearn.preprocessing import PowerTransformer

class Variable:

    def __init__(self,name,parent):
        
        self.name = name
        
        self.multiplier = 1.
        
    def update(self,expected,real):
        gam = self.parent.gamma
        
#        upd = ((real/expected-1) * gam)+1
#        self.multiplier = self.multiplier * upd    
        
        upd = (real - expected) * gam
        self.multiplier = self.multiplier + upd    
    
class Team:
    
    def __init__(self,name,parent):
        self.name = name
        self.parent = parent
        self.gamedata = parent.df[(parent.df.HOME==name) | (parent.df.HOME==name)]
        self.reset()
        
    def reset(self):
        self.home_attack = 1.
        self.home_defend = 1.
        self.away_attack = 1.
        self.away_defend = 1.
        self.attack = 1.
        self.defend = 1.
        
    def rank(self):
        #atta = (self.home_attack + self.away_attack) / 2
        #defe = (self.home_defend + self.away_defend) / 2
        atta = self.attack
        defe = self.defend
        compav =  self.parent.moving_ave/2
        rank = compav*(atta-defe)
        
        return rank
    
        
    def update(self,myscore,myexp,oppscore,oppexp):
        
        gam = self.parent.gamma

        myrealscore,myrealexp = self.parent.pt.transform(np.array([myscore,myexp]).reshape(1,2))[0]
        
        self.parent.variance += ((myrealscore-myrealexp)/myrealexp)**2
        self.parent.variance_n += 1
                
        self.parent.std = (self.parent.variance/self.parent.variance_n)**0.5
        
        upd = ((myscore/myexp-1) * gam)+1
        self.attack = self.attack * upd
        
        upd = ((oppscore/oppexp-1) * gam) + 1
        self.defend = self.defend * upd
        
        
    def homeupdate(self,myscore,myexp,oppscore,oppexp):
        
        gam = self.parent.gamma
        
        upd = ((myscore/myexp-1) * gam)+1
        self.home_attack = self.home_attack * upd
        
        upd = ((oppscore/oppexp-1) * gam) + 1
        self.home_defend = self.home_defend * upd
    
                
    def awayupdate(self,myscore,myexp,oppscore,oppexp):
        
        gam = self.parent.gamma
        
        upd = ((myscore/myexp-1) * gam)+1
        self.away_attack = self.away_attack * upd
        
        upd = ((oppscore/oppexp-1) * gam) + 1
        self.away_defend = self.away_defend * upd
        
        self.attack = (self.home_attack + self.away_attack) / 2
    
    def __repr__(self):
        return f'Attack: {self.attack} Defense: {self.defend}\n'

class Comp:
    
    def __init__(self,df,gamma=0.1,comp_gamma=0.02):
        
        self.pt = PowerTransformer()
        
        self.total_gamma = comp_gamma
        self.gamma = gamma
        
        self.df = df
        self.melted = pd.concat([df[['HOME','HOME SCORE']],df[['AWAY','AWAY SCORE']]],sort=True)
        
        self.variance = 0.
        self.variance_n = 0
        
        teams = set(df[['HOME','AWAY']].values.flatten())
            
        self.initialise_comp_averages()
        
        self.teams = {}
        
        for team in teams:
            self.teams[team] = Team(team,self)
            
    def initialise_comp_averages(self,end=None):
            
        df = self.df
        
        initial = df['TOTAL'].mean()
        
        self.moving_ave = initial
        self.mov_home_ave = df['HOME SCORE'].dropna().mean()
        self.mov_away_ave = df['AWAY SCORE'].dropna().mean()
            
    def update(self):
        
        self.error = 0.
        
        for r in self.df.dropna(subset=['TOTAL']).iterrows():
            i,row = r
            homename = row['HOME']
            awayname = row['AWAY']
            hometeam = self.teams[homename]    
            awayteam = self.teams[awayname]
            homescore = row['HOME SCORE']
            awayscore = row['AWAY SCORE']
            
            home_exp, away_exp = self.predict(homename,awayname)
            
            self.error += (home_exp-homescore)**2 + (away_exp-awayscore)**2
               
            hometeam.update(homescore,home_exp,awayscore,away_exp)
            awayteam.update(awayscore,away_exp,homescore,home_exp)
            
            self.update_comp_averages(homescore,home_exp,awayscore,away_exp)
            
    def update_comp_averages(self,h_s,hexp,a_s,aexp):
        
        gam = self.total_gamma
                
        upd = (h_s/hexp-1) * gam + 1
        self.mov_home_ave = self.mov_home_ave * upd
        
        upd = (a_s/aexp-1) * gam + 1
        self.mov_away_ave = self.mov_away_ave * upd
        
        self.moving_ave = self.mov_home_ave + self.mov_away_ave
                

    def predict(self,h,a):

        hometeam = self.teams[h]
        awayteam = self.teams[a]

        home_exp = self.mov_home_ave * hometeam.attack * awayteam.defend
        away_exp = self.mov_away_ave * awayteam.attack * hometeam.defend        
        
        return home_exp,away_exp
    
    def simulate(self,h_s,a_s,n=1000):
        
        h_s,a_s = self.pt.transform(np.array([h_s,a_s]).reshape(1,2))[0]
        
        h_std = abs(self.std * h_s)
        a_std = abs(self.std * a_s)
        
        h_s = np.random.normal(h_s,h_std,n).astype(int)
        a_s = np.random.normal(a_s,a_std,n).astype(int)
#        h_s = np.random.normal(h_s,h_std,n)
#        a_s = np.random.normal(a_s,a_std,n)
        
        games = zip(h_s,a_s)
        
        hwins = np.sum([1 for h,a in games if h>a])
        hchance = np.round(hwins/n,4)
        achance = np.round(1-hchance,4)
        
        return hchance, achance
    
    def rank_teams(self):
        
        teams = {k:v.rank() for k,v in self.teams.items()}
        
        return pd.DataFrame(data=teams.values(),columns=['Ranking'],index=teams.keys()).sort_values(by='Ranking',ascending=False)
    
    def get_home_factor(self,team,venue):
        
        filtered = self.df[self.df.HOME == team]
        
    
    def optimise_lr(self,lr=0.1,dg=0.02,tol=1e-6):
        
        self.gamma = lr
        
        count = 0
        
        self.update()
        
        err0 = self.error
        
        best = np.inf
        best_gamma = None
        
        while True:
            
            for team in self.teams.values():
                team.reset()
                
            self.gamma += dg
            
            self.gamma = min(self.gamma,0.5)
            self.gamma = max(self.gamma,0.01)
 
            self.update()
            
            if self.error < best:
                best = self.error
                best_gamma = self.gamma
                
            
            if abs((self.error - err0)/err0) < tol or count > 100:
                print(f'Done. Best Error was {best} with gamma {best_gamma}.')
                self.gamma = best_gamma
                break

        
            dE = (self.error-err0)
            dEdg = dE/dg
            
            dg = -(0.2*self.error)/dEdg * self.gamma
            
            err0 = self.error
            
            (self.error,err0,self.gamma,dg)
            
            count += 1