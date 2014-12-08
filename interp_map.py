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
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.path import Path
from matplotlib.patches import PathPatch
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

class MainDisplay(object):
    """docstring for MainDisplay"""
    def __init__(self, lllon=-180, lllat=-80, urlon=0, urlat=40, figsize=(11.7,8.3), fileshape='borders/COL_adm/COL_adm0'):
        super(MainDisplay, self).__init__()
        plt.clf()
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, axisbg='w', frame_on=False)
        self.anc_map = Basemap(projection = 'merc', llcrnrlon = lllon,
                                llcrnrlat = lllat, urcrnrlon = urlon,
                                urcrnrlat = urlat, resolution='h')
        self.anc_map.readshapefile(fileshape, 'borders', drawbounds=False)
    
    def draw(self, xi, yi, zi, x, y, z, coordinates, ancestry, shape_clip):
        """
        """
        norm = Normalize()
        self.anc_map.drawmapboundary(fill_color = 'white')
        self.anc_map.fillcontinents(color='#C0C0C0', lake_color='#7093DB')
        self.anc_map.drawcountries(
            linewidth=.75, linestyle='solid', color='#000073',
            antialiased=True,
            ax=self.ax, zorder=3)

        # contour plot
        con = self.anc_map.contourf(xi, yi, zi, zorder=5, cmap='RdPu', levels=np.arange(round(z.min()), round(z.max()), 0.005))
        # clip the data so only display the data inside of the country
        for contour in con.collections:
            contour.set_clip_path(shape_clip)

        # scatter plot
        self.anc_map.scatter(
            coordinates['projected_lon'],
            coordinates['projected_lat'],
            color='#545454',
            edgecolor='#ffffff',
            alpha=.75,
            s=30 * norm(ancestry),
            cmap='RdPu',
            ax=self.ax,
            vmin=zi.min(), vmax=zi.max(), zorder=5)

        # add colour bar and title
        cbar = self.anc_map.colorbar()

def process_shapefile(filename_shp, my_map, ax):
    sf = shapefile.Reader(filename_shp)

    for shape_rec in sf.shapeRecords():
        vertices = []
        codes = []
        lons,lats = zip(*shape_rec.shape.points)
        pts = np.array(my_map(lons, lats)).T
        prt = list(shape_rec.shape.parts) + [len(pts)]
        for i in range(len(prt) - 1):
            for j in range(prt[i], prt[i+1]):
                vertices.append((pts[j][0], pts[j][1]))
            codes += [Path.MOVETO]
            codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
            codes += [Path.CLOSEPOLY]
        clip = Path(vertices, codes)
        clip = PathPatch(clip, transform=ax.transData)
        print clip

    return clip




def main(filename_coord, filename_anc, column):
    """
    """
    SHAPEFILE = 'borders/COL_adm/COL_adm0'
    # set up plot
    lllon = -180
    lllat = -80
    urlon = 0
    urlat = 40
    display = MainDisplay(lllon, lllat, urlon, urlat, fileshape=SHAPEFILE)
    # load ancestry and location data
    map_data = MapData(filename_coord, filename_anc, columns, nrows=5000)
    map_data.get_coordinates()
    map_data.get_ancestry_average_by_coordinates(columns[1])

    map_data.project_coordinates(display.anc_map)
    xi, yi, zi, x, y, z = map_data.interpolate()
    shape_clip = process_shapefile(SHAPEFILE, display.anc_map, display.ax)

    display.draw(xi, yi, zi, x, y, z, map_data.coordinates, map_data.ancestry_avg, shape_clip)
    
    plt.title("Mean Anc {}".format(column[1]))
    #plt.savefig("native_{}.png".format(column[1]), format="png", dpi=300, transparent=True)
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
    #columns = ['CODE', 'SangerM-Nahua', 'Can-Mixe', 'Can-Mixtec', 'Can-Zapotec', 'Can-Kaqchikel', 'Can-Embera',
    #'Can-Kogi', 'Can-Wayuu', 'Can-Aymara', 'Can-Quechua', 'SangerP-Quechua', 'Can-Chane', 'Can-Guarani',
    #'Can-Wichi', 'CEU', 'GBR','IBS', 'TSI', 'LWK', 'MKK', 'YRI', 'CDX', 'CHB', 'CHS', 'JPT', 'KHV', 'GIH']
    #columns = ['CODE', 'SangerM-Nahua', 'Can-Mixe', 'GBR','IBS', 'TSI']
    columns = ['CODE', 'Can-Kogi']
    main(coord, anc, columns)
