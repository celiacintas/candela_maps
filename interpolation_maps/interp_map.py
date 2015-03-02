#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mapdata import MapData
import shapefile
import argparse
import json
import sys

class MainDisplay(object):
    """In this class we have the reference to our display map and the method of how to draw it."""
    def __init__(self, lllon=-180, lllat=-80, urlon=0, urlat=40, figsize=(11.7,8.3), files_shape=['borders/COL_adm/COL_adm0']):
        super(MainDisplay, self).__init__()
        plt.clf()
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, axisbg='w', frame_on=False)
        self.anc_map = Basemap(projection = 'merc', llcrnrlon = lllon,
                                llcrnrlat = lllat, urcrnrlon = urlon,
                                urcrnrlat = urlat, resolution='h')
        map(lambda country_fileshape: self.anc_map.readshapefile(country_fileshape, 'borders', drawbounds=False, linewidth=0.8),
            files_shape)
    
    def draw(self, xi, yi, zi, x, y, z, coordinates, ancestry, shape_clip, level_min, level_max):
        """
        This methods display the ancestry data from the MapData class, in this
        method you can setup the color display and resolution of the map.
        """
        norm = Normalize()
        self.anc_map.drawmapboundary(fill_color = 'white')
        self.anc_map.fillcontinents(color='#C0C0C0', lake_color='#7093DB')
        self.anc_map.drawcountries(
            linewidth=.75, linestyle='solid', color='#000073',
            antialiased=True,
            ax=self.ax, zorder=3)
        
        # contour plot
        con = self.anc_map.contourf(xi, yi, zi, zorder=5, levels=np.arange(0.0, 1.1, .1)) #, cmap='jet',
                                    #levels=np.arange(0.0001, z.max(), 0.05), antialiased=True)
        # check alpha parameter for areas without data
        # TODO fix the levels .. hardcoded number for now 
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
            s=30, #* norm(ancestry),
            cmap='RdPu',
            ax=self.ax,
            vmin=zi.min(), vmax=zi.max(), zorder=5)

        # add colour bar
        

    def add_colorbar(self):
        cbar = self.anc_map.colorbar()

# TODO move this to an other module
def process_shapefile(filename_shp, my_map, ax):
    # http://basemaptutorial.readthedocs.org/en/latest/clip.html
    sf = shapefile.Reader(filename_shp) #nasty
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

    return clip

def main(filename_coord, filename_anc, column, shapefile, boundry_lines):
    """
    """
    # set up plot
    lllon = -120 
    lllat = -70
    urlon = -20
    urlat = 40
    display = MainDisplay(lllon, lllat, urlon, urlat, files_shape=shapefile)

    # TODO calculate this on the mapdata module
    level_min, level_max = -1.0, 70.2
    # load ancestry and location data
    for country, anc, boundry_rect in zip(shapefile, filename_anc, boundry_lines):
        map_data = MapData(filename_coord, anc, columns)
        map_data.get_coordinates()
        #map_data.get_ancestry_average_by_coordinates(columns[1])

        map_data.project_coordinates(display.anc_map, boundry_rect) # pass the rect of the country
        xi, yi, zi, x, y, z = map_data.interpolate(columns[1])
        shape_clip = process_shapefile(country, display.anc_map, display.ax)

        display.draw(xi, yi, zi, x, y, z, map_data.coordinates, map_data.df[columns[1]], shape_clip, level_min, level_max)
    display.add_colorbar()

    plt.title("Mean Anc {}".format(column[1]))
    #plt.savefig("native_{}.png".format(column[1]), format="png", dpi=300, transparent=True)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description= 'Interpolation Maps for ..')
    parser.add_argument("--coor", dest="coordinate", default=None,
                        help='Pass the path to the txt file with lat lon and code')
    parser.add_argument("--anc_name", dest="ancestry", default=None, nargs='+',
                        help='Pass the path to the txt file with ancestry and code')
    parser.add_argument("--cluster_percentage", dest="cluster", default='GBR',
                        help='Pass the path to the txt file with ancestry and code')
    parser.add_argument("--country", dest="country", default='Colombia', nargs='+',
                        help='Pass the name of the country you want to display')

    args = parser.parse_args()
    
    if args.coordinate:
        coord = args.coordinate
    if args.ancestry:
        anc = args.ancestry
    if args.country:
        countries = args.country
    else:
        parser.print_help()
        sys.exit(1)

    with open('data/border_data.json') as json_data:
        hardcoded_dic = json.load(json_data)
    
    """
    clusters availables from Camilo's file
    'SangerM-Nahua', 'Can-Mixe', 'Can-Mixtec', 'Can-Zapotec', 'Can-Kaqchikel', 'Can-Embera',
    'Can-Kogi', 'Can-Wayuu', 'Can-Aymara', 'Can-Quechua', 'SangerP-Quechua', 'Can-Chane', 'Can-Guarani',
    'Can-Wichi', 'CEU', 'GBR','IBS', 'TSI', 'LWK', 'MKK', 'YRI', 'CDX', 'CHB', 'CHS', 'JPT', 'KHV', 'GIH'
    """
    columns = ['CODE', args.cluster]
    shape_files = map(lambda country: hardcoded_dic[country]['file_shape'], countries)
    boundry_lines = map(lambda bound_rect: dict((k, hardcoded_dic[bound_rect][k]) for k in ('lat', 'lon')), countries)
    main(coord, anc, columns, shape_files, boundry_lines)
