#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib.mlab import griddata
#from scipy.interpolate import griddata, interp2d
from mpl_toolkits.basemap import Basemap, interp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib import cm
import shapefile
import argparse
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

    def load_coordinates(self, filename):
        """the file has to be CODE Lat Lon """
        df = pd.read_csv(filename, sep='\t')

        return df

    def load_ancestry(self, filename, columns, nrows):
        """the file has to be CODE anc_0 anc_1 anc_2""""
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
        """give the coordinates to do the mesh (can't be duplicate data)"""
        self.coordinates = self.df[['Lat', 'Lon']].drop_duplicates()


    def project_coordinates(self, m):
        self.coordinates['projected_lon'], self.coordinates['projected_lat'] = m(*(self.coordinates['Lon'].values, self.coordinates['Lat'].values))

    def interpolate(self, numcols=1000, numrows=1000):
        xi = np.linspace(self.coordinates['projected_lon'].min(), self.coordinates['projected_lon'].max(), numcols)
        yi = np.linspace(self.coordinates['projected_lat'].min(), self.coordinates['projected_lat'].max(), numrows)
        xi, yi = np.meshgrid(xi, yi)
        # interpolate
        x, y, z = self.coordinates['projected_lon'].values, self.coordinates['projected_lat'].values, self.ancestry_avg
        zi = griddata(x, y, z, xi, yi)

        return xi, yi, zi, x, y, z


# TODO all this should move to a Map or View class
def create_map(lllon, lllat, urlon, urlat):
    """
    """
    map_ = Basemap(
        projection = 'merc',
        llcrnrlon = lllon, llcrnrlat = lllat, urcrnrlon = urlon, urcrnrlat = urlat,
        resolution='h')
    shp_info = map_.readshapefile('borders/COL_adm/COL_adm0', 'borders', drawbounds=True)

    return map_
    
def process_shapefile(filename_shp, my_map):
    r = shapefile.Reader(filename_shp)
    shapes = r.shapes()
 
    records = r.records()
     
    for record, shape in zip(records,shapes):
        lons,lats = zip(*shape.points)
        data = np.array(my_map(lons, lats)).T
        print data
        if len(shape.parts) == 1:
            segs = [data,]
        else:
            segs = []
            for i in range(1,len(shape.parts)):
                index = shape.parts[i-1]
                index2 = shape.parts[i]
                segs.append(data[index:index2])
            segs.append(data[index2:])
     
    lines = LineCollection(segs,antialiaseds=(1,))
    lines.set_facecolors(cm.jet(np.random.rand(1)))
    lines.set_edgecolors('k')
    lines.set_linewidth(0.1)

    return lines

def draw(map_, xi, yi, zi, x, y, z, coordinates, ancestry, ax):
        """
        """
        norm = Normalize()
        map_.drawmapboundary(fill_color = 'white')
        map_.fillcontinents(color='#C0C0C0', lake_color='#7093DB')
        map_.drawcountries(
            linewidth=.75, linestyle='solid', color='#000073',
            antialiased=True,
            ax=ax, zorder=3)

        # contour plot
        con = map_.contourf(xi, yi, zi, zorder=4, cmap='RdPu', levels=np.arange(round(z.min()), round(z.max()), 0.1))
        # scatter plot
        map_.scatter(
            coordinates['projected_lon'],
            coordinates['projected_lat'],
            color='#545454',
            edgecolor='#ffffff',
            alpha=.75,
            s=30 * norm(ancestry),
            cmap='RdPu',
            ax=ax,
            vmin=zi.min(), vmax=zi.max(), zorder=5)

        # add colour bar and title
        cbar = map_.colorbar()


def main(filename_coord, filename_anc, columns):
    """
    """
    # set up plot
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)

    map_data = MapData(filename_coord, filename_anc, columns, nrows=5000)
    map_data.get_coordinates()
    map_data.get_ancestry_average_by_coordinates(columns[1])

    # define map extent
    lllon = -180
    lllat = -80
    urlon = 0
    urlat = 40

    my_map = create_map(lllon, lllat, urlon, urlat)
    print my_map
    map_data.project_coordinates(my_map)
    xi, yi, zi, x, y, z = map_data.interpolate()
    draw(my_map, xi, yi, zi, x, y, z, map_data.coordinates, map_data.ancestry_avg, ax)
   
    shape_lines = process_shapefile("borders/COL_adm/COL_adm0", my_map)
    ax.add_collection(shape_lines)

    plt.title("Mean Anc {}".format(columns[1]))
    plt.savefig("native.png", format="png", dpi=300, transparent=True)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description= 'Interpolation Maps for ..')
    parser.add_argument("--coor", dest="coordinate", default=None,
                        help='Pass the path to the txt file with lat lon and code')
    parser.add_argument("--anc", dest="ancestry", default=None,
                        help='Pass the path to the txt file with ancestry and code')

    args = parser.parse_args()
    
    if args.coordinate:
        coord = args.coordinate
    if args.ancestry:
        anc = args.ancestry
    else:
        parser.print_help()
        sys.exit(1)
    
    #TODO pass by parameter
    columns = ['CODE', 'GBR']
    main(coord, anc, columns)