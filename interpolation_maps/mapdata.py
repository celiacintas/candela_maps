#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib.mlab import griddata
from scipy.interpolate import Rbf
from sklearn.gaussian_process import GaussianProcess
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import sys

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
        self.df.to_csv('jijiji.txt', index=False)

    def load_coordinates(self, filename):
        """the file has to be CODE Lat Lon """
        df = pd.read_csv(filename, sep='\s+')

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
        self.coordinates = self.df[['Lat', 'Lon']]#.drop_duplicates()

    def project_coordinates(self, m, boundry_country):
        self.coordinates['projected_lon'], self.coordinates['projected_lat'] = m(*(self.coordinates['Lon'].values, self.coordinates['Lat'].values))
        self.tmp_bound_lon, self.tmp_bound_lat = m(*(boundry_country['lon'], boundry_country['lat']))


    def interpolate(self, name_ancestry, numcols=50, numrows=50):
        """
        Take the boundry rect projected of the country to generate a meshgrid 
        """
       
        xi = np.linspace(min(self.tmp_bound_lon), max(self.tmp_bound_lon), numcols) #nasty fix
        yi = np.linspace(min(self.tmp_bound_lat), max(self.tmp_bound_lat), numrows)
        xi, yi = np.meshgrid(xi, yi)

        # interpolate 
        # TODO search for other interpolation types

        x, y, z = self.coordinates['projected_lon'].values, self.coordinates['projected_lat'].values, self.df[name_ancestry]
        print x.shape, y.shape, z.shape
        interp = Rbf(x, y, z, smooth=0.01, fuction='thin_plate')
        zi = interp(xi, yi)
        
        zi = np.clip(zi, a_min=0., a_max=1.)
        #print zi
        # Kriging interpolaiion
        """gp = GaussianProcess(regr='constant', corr='absolute_exponential',
                  theta0=0.01, nugget=0.082864)
        print gp.fit(X=np.column_stack([x,y]), y=z)
        rr_cc_as_cols = np.column_stack([xi.flatten(), yi.flatten()])
        zi = gp.predict(rr_cc_as_cols).reshape(yi.shape)
        
        # this is for selecting the best parameter
        from sklearn.grid_search import GridSearchCV

        gp = GaussianProcess()
        parameter_grid = {'theta0': np.logspace(-7, 0), 'nugget': np.logspace(-5, 3)}
        cv = GridSearchCV(gp, parameter_grid, scoring='mean_squared_error')
        cv.fit(X=np.column_stack([x,y]), y=z)
        gp_best = GaussianProcess(**cv.best_params_)
        gp_best.fit(X=np.column_stack([x,y]), y=z)
        print cv.best_params_
        """

        return xi, yi, zi, x, y, z


