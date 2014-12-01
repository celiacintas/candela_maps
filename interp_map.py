#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib.mlab import griddata
from mpl_toolkits.basemap import Basemap, interp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse


def load_data_excel(filename, columns):
	"""
	"""
	df = pd.read_excel(filename, sheetname='Sheet1')
	
	return df[columns]


def main(filename, columns):
	map_data = load_data_excel(filename, columns)
	print map_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description= 'Interpolation Maps for ..')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", dest="file", default=None,
                        help='Pass the path to the excel file with lat lon and %')
    args = parser.parse_args()
    
    if args.file:
        filename = args.file
    else:
        parser.print_help()
        sys.exit(1)
    
    #TODO pass by parameter
    columns = ['CODE', 'Lat', 'Lon', 'Native', 'European', 'African']
   	main(filename, columns)