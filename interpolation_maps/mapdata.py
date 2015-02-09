#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib.mlab import griddata
from scipy.interpolate import Rbf
from sklearn.gaussian_process import GaussianProcess

def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

class MapData(object):
    """
    On map data you can find the coordinates and the ancestry information
    ordered by code.
    """
    def __init__(self, filename_coor, filename_anc, columns, nrows=5000):
        """
        load the data files (coord and ancestry) and merge the data by code.
        """
        super(MapData, self).__init__()
        df_coordinates = self.load_coordinates(filename_coor)
        df_ancestry = self.load_ancestry(filename_anc, columns, nrows)
        self.df = pd.merge(df_coordinates, df_ancestry, on=['CODE'])

    def load_coordinates(self, filename):
        """the file has to be CODE Lat Lon """
        df = pd.read_csv(filename, sep='\t')

        return df

    def load_ancestry(self, filename, columns, nrows):
        """
        the file has to be CODE anc_0 anc_1 anc_2
        """
        df = pd.read_csv(filename, sep='\t', nrows=nrows)
        return df[columns]

    def get_ancestry_average_by_coordinates(self, name_ancestry):
        """
        In this method we take the average of each coordinate in the df
        """
        g = self.df.groupby(['Lat', 'Lon'])
        list_coord = list(g)
        self.ancestry_avg = np.array(map(lambda i: np.average(i[1][name_ancestry].values) * 100, list_coord))

    def get_coordinates(self):
        """
        give the coordinates to do the mesh (can't be duplicate data)
        """
        self.coordinates = self.df[['Lat', 'Lon']].drop_duplicates()

    def project_coordinates(self, m, boundry_country):
        self.coordinates['projected_lon'], self.coordinates['projected_lat'] = m(*(self.coordinates['Lon'].values, self.coordinates['Lat'].values))
        self.tmp_bound_lon, self.tmp_bound_lat = m(*(boundry_country['lon'], boundry_country['lat']))


    def interpolate(self, numcols=900, numrows=900):
        """
        Take the boundry rect projected of the country to generate a meshgrid 
        """
       
        xi = np.linspace(min(self.tmp_bound_lon), max(self.tmp_bound_lon), numcols) #nasty fix
        yi = np.linspace(min(self.tmp_bound_lat), max(self.tmp_bound_lat), numrows)
        xi, yi = np.meshgrid(xi, yi)

        # interpolate 
        # TODO search for other interpolation types
        x, y, z = self.coordinates['projected_lon'].values, self.coordinates['projected_lat'].values, self.ancestry_avg
        #interp = Rbf(x, y, z, function='linear', smooth=0.1)
        #zi = interp(xi, yi)
        
        # Kriging interpolation
        gp = GaussianProcess(theta0=10., thetaL=10., thetaU=100., nugget=0.01, verbose=True)
        gp.fit(X=np.column_stack([x,y]), y=z)
        rr_cc_as_cols = np.column_stack([xi.flatten(), yi.flatten()])
        zi = gp.predict(rr_cc_as_cols).reshape(yi.shape)

        return xi, yi, zi, x, y, z

