#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib.mlab import griddata

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

    def project_coordinates(self, m):
        self.coordinates['projected_lon'], self.coordinates['projected_lat'] = m(*(self.coordinates['Lon'].values, self.coordinates['Lat'].values))

    def interpolate(self, numcols=1000, numrows=1000):
        """
        Take the convex hull of all cordinates to generate a meshgrid
        """
        xi = np.linspace(self.coordinates['projected_lon'].min(), self.coordinates['projected_lon'].max(), numcols)
        yi = np.linspace(self.coordinates['projected_lat'].min(), self.coordinates['projected_lat'].max(), numrows)
        xi, yi = np.meshgrid(xi, yi)
        # interpolate
        x, y, z = self.coordinates['projected_lon'].values, self.coordinates['projected_lat'].values, self.ancestry_avg
        zi = griddata(x, y, z, xi, yi)

        return xi, yi, zi, x, y, z