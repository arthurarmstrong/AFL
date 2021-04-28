from datetime import datetime
import os, pandas as pd, numpy as np
from sklearn.preprocessing import PowerTransformer
from Ladder import SeasonPredicter
from venues import VenueAPI

class Variable:

    def __init__(self,name,parent):
        
        self.name = name
        self.parent = parent
        
        self.multiplier = 1.
        
    def update(self,expected,real):
        gam = self.parent.slow_gamma
        
        upd = ((real/expected-1) * gam)+1
        self.multiplier = self.multiplier * upd    
        
    def __repr__(self):
        return str(self.name)
    
class Team:
    
    def __init__(self,name,parent):
        self.name = name
        self.parent = parent
        
        self.home_attack = 1.
        self.home_defend = 1.
        self.away_attack = 1.
        self.away_defend = 1.
        self.attack = 1.
        self.defend = 1.
        self.home_attack_backup = 1.
        self.away_attack_backup = 1.
        self.home_defend_backup = 1.
        self.away_defend_backup = 1.
        self.attack_backup = 1.
        self.defend_backup = 1.
        
    def reset(self):
        self.home_attack = self.home_attack_backup
        self.home_defend = self.home_defend_backup
        self.away_attack = self.away_attack_backup
        self.away_defend = self.away_defend_backup
        self.attack = self.attack_backup
        self.defend = self.defend_backup
        
    def backup(self):
        self.home_attack_backup = self.home_attack
        self.away_attack_backup = self.away_attack
        self.home_defend_backup = self.home_defend
        self.away_defend_backup = self.away_defend
        self.attack_backup = self.attack
        self.defend_backup = self.defend
        
    def rank(self):
        #atta = (self.home_attack + self.away_attack) / 2
        #defe = (self.home_defend + self.away_defend) / 2
        atta = self.attack
        defe = self.defend
        compav =  self.parent.moving_ave
        rank = compav*(atta-defe)
        
        return rank
    
        
    def update(self,myscore,myexp,oppscore,oppexp):
        
        gam = self.parent.gamma

#        myrealscore = self.parent.pt.transform(np.array([myscore]).reshape(1,1))[0]-self.parent.normalize_offset
#        myrealexp = self.parent.pt.transform(np.array([myexp]).reshape(1,1))[0]-self.parent.normalize_offset

        self.parent.variance += (myscore-myexp)**2
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
    
    def __init__(self,df,gamma=0.1,slow_gamma=0.02,normalize_offset=5):
        
        self.pt = PowerTransformer()
        
        self.venues = VenueAPI()
        
        self.normalize_offset=normalize_offset
        
        self.slow_gamma = slow_gamma
        self.gamma = gamma
        
        self.df = df
        self.untransformed = df.copy()
        
        self.upcoming_fixtures = df[self.df.index>datetime.now()]
        
        self.transform_scores()
        #self.melted = pd.concat([df.NONVARIABLE[['HOME','HOME SCORE']],df.NONVARIABLE[['AWAY','AWAY SCORE']]],sort=True)
        
        self.variance = 0.
        self.variance_n = 0
        
        teams = set(df[pd.MultiIndex.from_product([['NONVARIABLE'],['HOME','AWAY']])].values.flatten())
            
        self.initialise_comp_averages()
        
        self.teams = {}
        
        self.create_variables()
        
        for team in teams:
            self.teams[team] = Team(team,self)
            
        self.season_predictor = SeasonPredicter(self)
            
    def transform_scores(self):
        scores = self.df[[('NONVARIABLE','HOME SCORE'),('NONVARIABLE','AWAY SCORE')]].values.reshape(-1,1)
        self.pt.fit_transform(scores)
        self.df[('NONVARIABLE','HOME SCORE')] = self.pt.transform(self.df[('NONVARIABLE','HOME SCORE')].values.reshape(-1,1))+5
        self.df[('NONVARIABLE','AWAY SCORE')] = self.pt.transform(self.df[('NONVARIABLE','AWAY SCORE')].values.reshape(-1,1))+5
        
        self.df[('NONVARIABLE','TOTAL')] = self.df.loc[:,('NONVARIABLE','HOME SCORE')] + self.df.loc[:,('NONVARIABLE','AWAY SCORE')]
    
            
    def initialise_comp_averages(self,end=None):
            
        df = self.df
        
        initial = df[[('NONVARIABLE','TOTAL')]].mean().NONVARIABLE.TOTAL/2
        
        self.moving_ave = initial
        self.mov_home_ave = df[[('NONVARIABLE','HOME SCORE')]].dropna().mean().NONVARIABLE['HOME SCORE']
        self.mov_away_ave = df[[('NONVARIABLE','AWAY SCORE')]].dropna().mean().NONVARIABLE['AWAY SCORE']
            
    def update(self):

        self.error = 0.
        
        for i,row in self.df.dropna(subset=[('NONVARIABLE','TOTAL')]).iterrows():
            
            homevars,awayvars = self.get_active_vars(row)
            
            homename = row[('NONVARIABLE','HOME')]
            awayname = row[('NONVARIABLE','AWAY')]
            hometeam = self.teams[homename]    
            awayteam = self.teams[awayname]
            homescore = row[('NONVARIABLE','HOME SCORE')]
            awayscore = row[('NONVARIABLE','AWAY SCORE')]
            
            home_exp, away_exp = self.predict(homename,awayname,homevars,awayvars)
            
            self.error += (home_exp-homescore)**2 + (away_exp-awayscore)**2
               
            hometeam.update(homescore,home_exp,awayscore,away_exp)
            awayteam.update(awayscore,away_exp,homescore,home_exp)
            
            #Finally, update variables (e.g. based on distance travelled)
            for var in homevars:
                var.update(home_exp,homescore)
            for var in awayvars:
                var.update(away_exp,awayscore)
            
            self.update_comp_averages(homescore,home_exp,awayscore,away_exp)
            
    def update_comp_averages(self,h_s,hexp,a_s,aexp):
        
        gam = self.slow_gamma
                
        upd = (h_s/hexp-1) * gam + 1
        self.mov_home_ave = self.mov_home_ave * upd
        
        upd = (a_s/aexp-1) * gam + 1
        self.mov_away_ave = self.mov_away_ave * upd
        
        realtot,exptot = h_s+a_s,hexp+aexp
        upd = (realtot/exptot-1) * gam + 1
        self.moving_ave = self.moving_ave * upd

    def predict(self,h,a,homevars,awayvars):

        hometeam = self.teams[h]
        awayteam = self.teams[a]

        home_exp = self.moving_ave * hometeam.attack * awayteam.defend
        for var in homevars:
            home_exp *= var.multiplier

        away_exp = self.moving_ave * awayteam.attack * hometeam.defend
        for var in awayvars:
            away_exp *= var.multiplier
        
        return home_exp,away_exp
    
    def monte_carlo(self,match,n=1000,invert=True):
        
        home,away = match.NONVARIABLE.HOME, match.NONVARIABLE.AWAY
        homevars, awayvars = self.get_active_vars(match)
        
        h_exp,a_exp = [np.round(x,3) for x in self.predict(home,away,homevars,awayvars)]
        
        h_std = self.std
        a_std = self.std
        
        h_s = np.random.normal(h_exp,h_std,n)
        a_s = np.random.normal(a_exp,a_std,n)
        
        if invert:
            h_s = self.pt.inverse_transform((h_s-self.normalize_offset).reshape(-1,1)).astype(int).flatten()
            a_s = self.pt.inverse_transform((a_s-self.normalize_offset).reshape(-1,1)).astype(int).flatten()

            h_exp = self.pt.inverse_transform(np.array(h_exp-self.normalize_offset).reshape(1,1))[0]
            a_exp = self.pt.inverse_transform(np.array(a_exp-self.normalize_offset).reshape(1,1))[0]
        
        return h_s,a_s,h_exp,a_exp
        
    
    def simulate(self,match,n=1000,scale=100,invert=True):
        
        h_s, a_s, h_exp, a_exp = self.monte_carlo(match,n=n,invert=invert)
        
        games = zip(h_s,a_s)
        
        hwins = np.sum([1 for h,a in games if h>a])
        
        hchance = np.round(hwins/n*scale,2)
        achance = np.round((scale-hchance),2)
        
        return hchance, achance, h_exp,a_exp
    
    def create_variables(self):
        fil = self.df.filter(like='VARIABLES')
        var_names = fil.columns.get_level_values(1).unique()
        
        self.variables = {x:Variable(x,self) for x in var_names}
        
    def get_active_vars(self,loc):
        variables = []
        for prefix in ['HOME','AWAY']:
            active = loc[f'{prefix}_VARIABLES']
            active = active.loc[active==1]
            var_names = active.index.unique()
            variables.append( [self.variables[x] for x in var_names])
        return variables
    
    def rank_teams(self):
        
        teams = {k:v.rank() for k,v in self.teams.items()}
        
        return pd.DataFrame(data=teams.values(),columns=['Ranking'],index=teams.keys()).sort_values(by='Ranking',ascending=False) 
    
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