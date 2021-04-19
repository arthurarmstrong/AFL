from geopy.geocoders import Nominatim
from geopy.distance import distance
import pickle, pandas as pd

class VenueAPI:
    
    def __init__(self):
        
        with open('TeamLocations','rb') as f:
            self.home_venues = pickle.load(f)
        with open('StadiumData','rb') as f:
            self.stadium_data = pickle.load(f)
        
        self.geolocater = Nominatim(user_agent='appy')


    def get_distance(self,team,venue):
        
        if self.stadium_data[venue]['Location'] is None:
            return 0
        
        teamcoords = self.home_venues[team]['Coords'].point
        venuecoords = self.stadium_data[venue]['Location'].point   
        return distance(teamcoords,venuecoords).km
    
    def add_team(self,team,city,save=False):
        geolocation = self.geolocater.geocode(city)
        self.home_venues[team] = {'City':city,'Coords':geolocation}
        
        if save:
            with open('TeamLocations','wb') as f:
                pickle.dump(self.home_venues,f)
                
    def get_series(self,df,as_category=True):
        """
        Take a DataFrame object and return a Series with the distance for each game
        """
        
        for prefix in ['HOME','AWAY']:
            distances = [self.get_distance(data.NONVARIABLE[prefix],data.NONVARIABLE.VENUE) for key, data in df.iterrows()]
            
            travel = pd.Series(data=distances,index=df.index)
            
            if as_category:
                m1 = travel > 1000
                m2 = travel <= 1000
                m3 = travel <= 50
                m4 = travel == 0
                
                for i,mask in enumerate([m1,m2,m3,m4]):
                    travel.loc[mask] = i
                travel = travel.astype(int)
                
                dummies = pd.get_dummies(travel,prefix=f'TRAVEL_DISTANCE')
                dummies.columns = pd.MultiIndex.from_product([[f'{prefix}_VARIABLES'],dummies.columns])
                
                df = pd.concat([df,dummies],axis=1)
                
        return df