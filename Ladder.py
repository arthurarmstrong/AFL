from datetime import datetime
import pandas as pd
import numpy as np, sys, pickle
from itertools import cycle
import matplotlib.pyplot as plt

colormap = {'Geelong': ((0, 51, 102),(211,211,211)) ,
 'Western Bulldogs': ((28, 46, 113),(164, 174, 181), (190, 0, 39)) ,
 'Essendon': ((0, 0, 0),(204, 32, 49)) ,
 'Sydney': ((216, 19, 26),(198, 210, 230)) ,
 'North Melbourne': ((19, 56, 143),(198, 210, 230)),
 'Richmond': ((0, 0, 0),(255, 211, 0)) ,
 'Hawthorn': ((77, 32, 4), (250, 191, 22)),
 'Fremantle': ((37, 5, 83), (164, 174, 181)),
 'Brisbane': ((0, 84, 164),(255, 225, 155),(253, 191, 87),(165, 0, 68), (124, 0, 62)),
 'Adelaide': ((0, 43, 92),(254, 209, 2), (50, 97, 156), (226, 25, 55)),
 'West Coast': ((0, 45, 136),(255,215,0)),
 'Gold Coast': ((255, 224, 16),(185, 10, 52),(233, 42, 31),(241, 93, 66), (243, 119, 85),(6, 129, 194)),
 'GWS': ((244, 122, 26),(164, 174, 181)) ,
 'St Kilda': ((0, 0, 0),(237, 27, 47),(164, 174, 181)),
 'Port Adelaide': ((0, 128, 131), (16, 12, 7),(164, 174, 181)),
 'Melbourne': ((7, 9, 45),(205, 26, 46)),
 'Collingwood': ((0, 0, 0),(211,211,211)),
 'Carlton':((0, 51, 102),(198, 210, 230))
     }

def cm_gen(team):
    
    return cycle(colormap[team])
    

class SeasonPredicter:
    
    def __init__(self,comp):
        self.comp = comp
        df = comp.untransformed
        
        self.this_season = df[df.index.year == datetime.now().year]
        self.this_season['PLAYED'] = self.this_season.index < datetime.now()
        
        self.separate_played_unplayed()
        
        self.gfwinners = []
        self.gfparticipants = []
        
    def separate_played_unplayed(self):
        
        self.played = self.this_season.loc[self.this_season.PLAYED].NONVARIABLE
        self.unplayed = self.this_season.loc[~self.this_season.PLAYED]
        self.played_ladder = self.build_ladder(self.played)
                
        
    def build_ladder(self,df):
        
        homewin = (df['HOME SCORE'] > df['AWAY SCORE'])
        awaywin = ~homewin
        draw = (df['HOME SCORE'] == df['AWAY SCORE'])
        df['HOME_POINTS'] = homewin*4 + draw*2
        df['AWAY_POINTS'] = awaywin*4 + draw*2
        
        pts = df.groupby('HOME')['HOME_POINTS'].sum() + df.groupby('AWAY')['AWAY_POINTS'].sum() 
        pts = pts.rename('Points').astype(int)
        f = df.groupby('HOME')['HOME SCORE'].sum() + df.groupby('AWAY')['AWAY SCORE'].sum()
        f = f.rename('For').astype(int)
        a = df.groupby('HOME')['AWAY SCORE'].sum() + df.groupby('AWAY')['HOME SCORE'].sum()
        a = a.rename('Against').astype(int)
        perc = f/a
        perc = perc.rename('%')
        
        ladder = pd.concat([pts,f,a,perc],axis=1)
        ladder = ladder.sort_values(by=['Points','%'],ascending=False)
        return ladder
    
    def forecast(self,forecast_to_grand_final=False,n=1000):
        for_predictions = []
        against_predictions = []
        win_predictions = []
        draw_predictions = []
        
        finishing_positions = []
        
        self.gfwinners = []
        
        for i,match in self.unplayed.iterrows():
            home, away = match.NONVARIABLE.HOME, match.NONVARIABLE.AWAY
            h_s,a_s, _, _ = self.comp.monte_carlo(match,n=n)
            h_s = pd.Series(h_s,name=home)
            a_s = pd.Series(a_s,name=away)
            
            for_predictions.append(h_s)
            for_predictions.append(a_s)
            
            against_predictions.append(a_s.rename(home))
            against_predictions.append(h_s.rename(away))
            
            draw_predictions.append((h_s == a_s).rename(home) * 2)
            draw_predictions.append((h_s == a_s).rename(away) * 2)
            
            win_predictions.append((h_s>a_s).rename(home) * 4)
            win_predictions.append((h_s<a_s).rename(away) * 4)

        #Each column is a team playing in a game simulated 
        self.for_predictions = pd.concat(for_predictions,axis=1)
        self.against_predictions = pd.concat(against_predictions,axis=1)
        self.win_predictions = pd.concat(win_predictions,axis=1)
        self.draw_predictions = pd.concat(draw_predictions,axis=1)
        
        #Each row is a simulation of the season
        for i,season_sim in self.win_predictions.iterrows():
            wins = season_sim.groupby(season_sim.index).sum().rename('Points')
            pts_for = self.for_predictions.iloc[i].groupby(season_sim.index).sum().rename('For')
            pts_agst = self.against_predictions.iloc[i].groupby(season_sim.index).sum().rename('Against')
            ladder = pd.concat([wins,pts_for,pts_agst],axis=1)
            ladder = ladder+self.played_ladder
            ladder['%'] = ladder.For / ladder.Against
            ladder.sort_values(by=['Points','%'],inplace=True,ascending=False)
            ladder['RANK'] = [x+1 for x in range(len(ladder.index.values))]
            self.ladder = ladder
            
            if forecast_to_grand_final:
                self.forecast_afl_finals(ladder)
            
            finishing_positions.append(ladder.RANK)
        
        self.finishing_positions = pd.concat(finishing_positions,axis=1)
        
        self.frequency_count()
        
    def slow_forecast(self,forecast_to_grand_final=False,n=1000):
        
        finishing_positions = []
        
        self.gfwinners = []

        #save the current values of team ratings        
        [t.backup() for t in self.comp.teams.values()]
        
        gridshape = (n,T.unplayed.shape[0]*2)
        win_grid = np.zeros(gridshape)
        draw_grid = np.zeros(gridshape)
        for_grid = np.zeros(gridshape)
        against_grid = np.zeros(gridshape)
        col_labels = self.unplayed.NONVARIABLE[['HOME','AWAY']].stack().values
        
        for sim_no in range(n):
            
            match_index = 0
            
            for i,match in self.unplayed.iterrows():
                home, away = match.NONVARIABLE.HOME, match.NONVARIABLE.AWAY
                pred = self.comp.monte_carlo(match,1)
                h_s,a_s,h_exp,a_exp = pred
                
                h_s_t,a_s_t, h_exp_t, a_exp_t = self.comp.pt.transform(pred).flatten()+self.comp.normalize_offset
                
                win_grid[sim_no,match_index] = 4 if h_s > a_s else 0
                draw_grid[sim_no,match_index] = 2 if h_s == a_s else 0
                for_grid[sim_no,match_index] = h_s
                against_grid[sim_no,match_index] = a_s
                
                self.comp.teams[home].update(h_s_t,h_exp_t,a_s_t,a_exp_t)
                                
                match_index += 1
                
                win_grid[sim_no,match_index] = 4 if a_s > h_s else 0
                draw_grid[sim_no,match_index] = 2 if h_s == a_s else 0
                for_grid[sim_no,match_index] = a_s
                against_grid[sim_no,match_index] = h_s
                
                self.comp.teams[away].update(a_s_t,a_exp_t,h_s_t,h_exp_t)
                
                match_index += 1
                
            #put the ratings back where they were    
            [t.reset() for t in self.comp.teams.values()]

        #Each column is a team playing in a game simulated 
        self.for_predictions = pd.DataFrame(columns=col_labels,data=for_grid)
        self.against_predictions = pd.DataFrame(columns=col_labels,data=against_grid)
        self.win_predictions =  pd.DataFrame(columns=col_labels,data=win_grid)
        self.draw_predictions =  pd.DataFrame(columns=col_labels,data=draw_grid)
        
        #Each row is a simulation of the season
        for i,season_sim in self.win_predictions.iterrows():
            wins = season_sim.groupby(season_sim.index).sum().rename('Points')
            pts_for = self.for_predictions.iloc[i].groupby(season_sim.index).sum().rename('For')
            pts_agst = self.against_predictions.iloc[i].groupby(season_sim.index).sum().rename('Against')
            ladder = pd.concat([wins,pts_for,pts_agst],axis=1)
            ladder = ladder+self.played_ladder
            ladder['%'] = ladder.For / ladder.Against
            ladder.sort_values(by=['Points','%'],inplace=True,ascending=False)
            ladder['RANK'] = [x+1 for x in range(len(ladder.index.values))]
            self.ladder = ladder
            
            if forecast_to_grand_final:
                self.forecast_afl_finals(ladder)
            
            finishing_positions.append(ladder.RANK)
        
        self.finishing_positions = pd.concat(finishing_positions,axis=1)
        
        self.frequency_count()
        
    def forecast_afl_finals(self,simladder):
        
        """
        Take a simulated ladder from the forecast function and use it to simulate the finals series
        """
                        
        (qf1winner, qf1loser), (qf2winner, qf2loser), (ef1winner,ef1loser),(ef2winner,ef2loser) = self.forecast_final_stage(simladder)
        
        pairs = [(qf1loser,ef1winner),(qf2loser,ef2winner)]
        
        (sf1winner, _), (sf2winner, _) = self.forecast_final_stage(simladder,pairs)
        
        pairs = [(qf1winner,sf1winner),(qf2winner,sf2winner)]
        
        (pf1winner, _), (pf2winner, _) = self.forecast_final_stage(simladder,pairs=pairs)
        
        pairs = [((pf1winner,pf2winner))]        
        
        [(gfwinner,_)] = self.forecast_final_stage(simladder,pairs=pairs)
        
        gfwinner = simladder.loc[simladder.RANK==gfwinner].index[0]
        pf1winner = simladder.loc[simladder.RANK==pf1winner].index[0]
        pf2winner = simladder.loc[simladder.RANK==pf2winner].index[0]
        
        self.gfwinners.append(gfwinner)
        self.gfparticipants.extend([pf1winner,pf2winner])
        
#        if len(self.gfwinners) % 100 == 0:
        sys.stdout.write('\r' + f'Season simulated {len(self.gfwinners)} times.')
        sys.stdout.flush()
        
    
    def forecast_final_stage(self,simladder,pairs=[(1,4),(2,3),(5,8),(6,7)]):
        """
        Inputs:
            Ladder - The current state of a ladder
            
            Pairs - The teams who will face off against one another.
        """
        
        matches = []
        winners = []
        losers = []
        
        for h,a in pairs:
            home,away = simladder.loc[simladder.RANK==h].index[0], simladder.loc[simladder.RANK==a].index[0]
            
            match_venue = self.comp.venues.get_closest_stadium(home)
            
            matchdate = datetime(2021,9,3,15,0)
            
            matches.append({
                            'DATE':matchdate,
                            'HOME':home,
                            'AWAY':away,
                            'VENUE':match_venue,
                            'HOME SCORE':np.nan,
                            'AWAY SCORE':np.nan
                            })
    
        matches = pd.DataFrame(data=matches).set_index('DATE')
        matches.columns = pd.MultiIndex.from_product([['NONVARIABLE'],matches.columns])
        
        matches = self.comp.venues.get_series(matches)
        
        for ind,match in matches.iterrows():
            home,away = match.NONVARIABLE.HOME, match.NONVARIABLE.AWAY
            hwin, awin, _, _ = self.comp.simulate(match,n=1,scale=1)
            if hwin:
                winners.append(simladder.loc[home].RANK)
                losers.append(simladder.loc[away].RANK)
            else:
                winners.append(simladder.loc[away].RANK)
                losers.append(simladder.loc[home].RANK)
            
        return zip(winners,losers)
            
    def frequency_count(self):
        counts = []
        for ind,team_finishes in self.finishing_positions.iterrows():
            val_count = pd.value_counts(team_finishes) / team_finishes.shape[0]
            counts.append(val_count.rename(ind))
        
        self.fullseason = pd.concat(counts,axis=1).fillna(0)
        self.cumulative_ladder = self.fullseason.cumsum()
        self.cumulative_ladder.to_pickle('CumulativeLadder')
                
    def cumulative_rank(self,team,rank):
        return self.cumulative_ladder[team].loc[rank]
    
    def regular_season_histogram(self):
        axs = self.finishing_positions.T.hist(figsize=(15,15),bins=range(1,20),density=True)
        
        for a in axs.flatten():
            team = a.title.get_text()
            
            if not team: continue
        
            a.add_patch(plt.Rectangle((0,0),9,1,alpha=0.1,color='red'))
        
            cmgen = cm_gen(team)
            for p in a.patches:
                color = next(cmgen)
                color = [x/255 for x in color]
                p.set_color(color)
            
            a.minorticks_on()
        
        return a.figure
    
    def to_make_finals(self):
        return 1/self.cumulative_ladder.loc[8].rename('To Make Finals').sort_values(ascending=False)
    
    def to_miss_finals(self):
        return 1/(1-self.cumulative_ladder.loc[8]).rename('To Miss Finals').sort_values(ascending=False)
    
    def to_win_gf(self):
        
        wingf = pd.value_counts(self.gfwinners)
        wingf /= wingf.sum()
        wingf.rename('Chance of Winning The Grand Final',inplace=True)
                
        return wingf
        
    def to_make_gf(self):
        makegf = pd.value_counts(self.gfparticipants)
        makegf /= makegf.sum() / 2
        makegf.rename('Chance of Making The Grand Final',inplace=True)

        return makegf
    
def plot_odds(df):
    fig, ax = plt.subplots()
    ax.set_xlim(right=19)
    ax.set_ylim(top=df.max()+0.1)
    
    for i,(team,y) in enumerate(df.iteritems()):
        
        cm = colormap[team]
        numpatches = len(cm)
        pheight = y
        pmap = np.linspace(0,pheight,numpatches+1)
        patchbins = [(pmap[i],pmap[i+1]) for i,x in enumerate(pmap[:-1])]
        
        for ipatch, vertpatch in enumerate(patchbins):
            p = plt.Rectangle((i+0.25,vertpatch[0]),0.5,pheight/numpatches)
            p.set_color([x/255 for x in cm[ipatch]])
            ax.add_patch(p)
    
    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index,rotation=270)
    ax.set_title(df.name)
    
    return ax
        
        
class PrettyPredictions:
    
    def __init__(self,predictor):
        self.make_finals = predictor.to_make_finals()
        self.miss_finals = predictor.to_miss_finals()
        self.cumulative_ladder = predictor.cumulative_ladder
        season_hist = predictor.regular_season_histogram()
        season_hist.savefig('season.png')
        
        upcoming = predictor.unplayed.NONVARIABLE.reset_index()
        
        wins = (predictor.win_predictions+predictor.draw_predictions)/4
        
        match_ind = np.array([[i,i] for i in range(int((wins.shape[1]+1)/2))]).flatten()
        wins.columns = pd.MultiIndex.from_arrays([match_ind,wins.columns])
        
        for i,game in upcoming.iterrows():
            pred = wins.loc[:,i]
            
            home = pred[pred.columns[0]]
            away = pred[pred.columns[1]]
            
            hwin = home.sum()/home.count()
            awin = away.sum()/away.count()
            
            upcoming.at[i,'HOME CHANCE'] = hwin
            upcoming.at[i,'AWAY CHANCE'] = awin
        
        upcoming.drop(labels=['HOME SCORE','AWAY SCORE','TOTAL'],inplace=True,axis=1)
        
        self.upcoming = upcoming.set_index('DATE')
        
    def save(self):
        with open('predictions','wb') as f:
            pickle.dump(self,f)
    
if __name__ == '__main__':
    T = SeasonPredicter(k)
##    T.forecast(forecast_to_grand_final=True,n=10000)
    T.slow_forecast(forecast_to_grand_final=True,n=2000)
    
    PrettyPredictions(T).save()
