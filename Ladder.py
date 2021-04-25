from datetime import datetime
import pandas as pd
import numpy as np, sys
from itertools import cycle

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
    
    def forecast(self,forecast_to_grand_final=False):
        for_predictions = []
        against_predictions = []
        win_predictions = []
        draw_predictions = []
        
        finishing_positions = []
        
        self.gfwinners = []
        
        for i,match in self.unplayed.iterrows():
            home, away = match.NONVARIABLE.HOME, match.NONVARIABLE.AWAY
            h_s,a_s, _, _ = self.comp.monte_carlo(match)
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
        
        self.gfwinners.append(gfwinner)
        
        if len(self.gfwinners) % 100 == 0:
            sys.stdout.write(f'Season simulated {len(self.gfwinners)} times.')
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
        
        print(self.cumulative_ladder)
        
    def cumulative_rank(self,team,rank):
        return self.cumulative_ladder[team].loc[rank]
    
    def regular_season_histogram(self):
        axs = self.finishing_positions.T.hist(figsize=(15,15),bins=range(1,20),density=True)
        
        for a in axs.flatten():
            team = a.title.get_text()
            
            if not team: continue
        
            cmgen = cm_gen(team)
            for p in a.patches:
                color = next(cmgen)
                color = [x/255 for x in color]
                p.set_color(color)
            
            a.minorticks_on()
        
        return a.figure
    
if __name__ == '__main__':
    T = SeasonPredicter(k)
    T.forecast(forecast_to_grand_final=True)
    T.frequency_count()
