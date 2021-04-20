from datetime import datetime
import pandas as pd
        
class SeasonPredicter:
    
    def __init__(self,comp):
        self.comp = comp
        df = comp.untransformed
        this_season = df[df.index.year == datetime.now().year]
        self.played = this_season[this_season.index < datetime.now()].NONVARIABLE
        self.unplayed = this_season[this_season.index > datetime.now()]
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
    
    def forecast(self):
        for_predictions = []
        against_predictions = []
        win_predictions = []
        draw_predictions = []
        
        finishing_positions = []
        
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
            
            finishing_positions.append(ladder.RANK)
        
        self.finishing_positions = pd.concat(finishing_positions,axis=1)
            
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
        
if __name__ == '__main__':
    T = SeasonPredicter(k)
    T.forecast()
    T.frequency_count()