#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division 

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MainDisplay(object):
    """docstring for MainDisplay"""
    def __init__(self, figsize=(11.7,8.3)):
        super(MainDisplay, self).__init__()
        self.fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,wspace=0.15,hspace=0.05)
        self.ax = plt.subplot(111)
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.colors = self.get_colors(75) #convert to a parameter variable

    def onmove(self, event):
        """ """
        pass

    def get_colors(self, num_clusters):
        """ """
        self.colormap = plt.cm.gist_ncar
        
        return [self.colormap(i) for i in np.linspace(0, 0.9, num_clusters)]

    def draw_pie_charts(self, label, ratios=[0.4,0.3,0.3], X=0, Y=0, size = 500):
        """ """
        N = len(ratios)
        xy = []
        start = 0.
        for ratio in ratios:
            x = [0] + np.cos(np.linspace(2*np.pi*start,2*np.pi*(start+ratio), 30)).tolist()
            y = [0] + np.sin(np.linspace(2*np.pi*start,2*np.pi*(start+ratio), 30)).tolist()
            xy1 = zip(x,y)
            xy.append(xy1)
            start += ratio
     
        for i, xyi in enumerate(xy):
            self.ax.scatter([X],[Y] , marker=(xyi,0), s=size, facecolor=self.colors[i], alpha=0.9 )
            self.ax.annotate(label, xy= (X, Y) , bbox = dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))


class MapData(object):
    """docstring for MapData"""
    def __init__(self, filename_coordinates, filename_clusters=None, filename_individuals=None):
        super(MapData, self).__init__()
        # see is this variable is necesary
        self.df_coordinates = pd.read_csv(filename_coordinates, sep=' ')
        self.df_clusters = map(lambda s: s.strip(), open(filename_clusters).readlines())
        self.df_individuals = pd.read_csv(filename_individuals, sep='\t', header=None)
        self.populations = self.get_values('Population')
        self.longitude = self.get_values('Longitude')
        self.latitude = self.get_values('Latitude')
        self.populations_ind = self.get_populations_individuals()
        
    def get_values(self, valuename):
        """docstring for get_values"""
        return self.df_coordinates[valuename].values

    def get_populations_individuals(self):
        """Get name and number of individuals in that population"""
        self.df_individuals.columns = ['ind', 'pop']
        groups = self.df_individuals.groupby('pop')

        return dict(map(lambda p: (p, [groups.get_group(p).shape[0], list(groups.get_group(p)['ind'].values)]), self.populations))


    def get_ratios(self, population):
        """Get the proportions of each cluster in one population"""
        clusters = self.get_clusters(population)
        total_individuals = self.populations_ind[population][0]

        return map(lambda p: p/total_individuals, clusters.values())


    def get_clusters(self, population):
        """distribute each individual of a population in the clusters"""
        clusters = dict(zip(range(1, 76), [0]*75))
        individuals = self.populations_ind[population][1] # This is the list of the indiviudals in the group
        for individual in individuals:
            for c in self.df_clusters:
                if c.find(individual) != -1:
                    clusters[int(c.split(' ')[0])] += 1
                    # gete number of cluster and add 1
        print population, clusters
        
        return clusters


class Map(Basemap):
    """docstring for Map"""
    def __init__(self, ax):
        super(Map, self).__init__(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
                llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='i', ax=ax)
        #pass arg by parameter so they can change proj and other things
    
    def draw(self, color='#9A9595', alpha=0.1):
        """ """
        #map_.bluemarble()
        #map_.shadedrelief()
        self.drawcoastlines()
        self.drawcountries()
        self.fillcontinents(color=color, alpha=alpha)
        self.drawmapboundary()

        self.drawmeridians(np.arange(0, 360, 30))
        self.drawparallels(np.arange(-90, 90, 30))



        
def main():
    """
    """
    FILENAME_COR = 'data/don_coordinates.txt'
    FILENAME_CLUS = 'data/CandelaFStree.populations.indlist.txt'
    FILENAME_IND_POP = 'data/candela_main.idfile.txt'
    
    my_display = MainDisplay()
    my_data = MapData(FILENAME_COR, FILENAME_CLUS, FILENAME_IND_POP)
    my_map = Map(my_display.ax)
    my_map.draw()
    
    my_data.get_clusters(my_data.populations_ind.keys()[0])
    x,y = my_map(my_data.longitude, my_data.latitude)
    
    map(lambda p: my_display.draw_pie_charts(p[0], ratios= my_data.get_ratios(p[0]), X=p[1], Y=p[2]), zip(my_data.populations, x, y))

    
    plt.show()

if __name__ == '__main__':
    main()
