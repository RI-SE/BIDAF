#!/usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt

from featurename import filter_names_scale, filter_names_mode
from causal_model import CausalModel
from higherorder import HigherOrderModel
#from AnomalyModel import AnomalyModel
from cluster_model import ClusterModel
from entity_visualizer import EntityVisualizer
from graph_visualizer import GraphVisualizer
from TimeSeriesVisualizer import TimeSeriesVisualizer
from ScatterPlotVisualizer import ScatterPlotVisualizer, anomaly_color, cluster_color
from datasourcefiledialog import FileDialog
from bidaf_framework import BidafFramework

models = [CausalModel, HigherOrderModel, ClusterModel] # , AnomalyModel

visualizers = [EntityVisualizer, GraphVisualizer, TimeSeriesVisualizer, ScatterPlotVisualizer]

settings = {'time_attribute':False,
            'entity_attribute':False,
            'index_attribute': False,
            'class_attribute':'ClusterIndex',
            'class_color':cluster_color,
            #'anomaly_attribute':'deviation',
            #'anomaly_color':anomaly_color,
            #'anomaly_threshold':0.5,
            'anomaly_attribute':'ClusterAnomaly',
            'anomaly_color':anomaly_color,
            'anomaly_threshold':0.0,
            'citest':'gauss',
            'significance':0.01,
            'tolerance':0.10,
            'data_source_name':False}

bidaf = None
datasource = None

def launch_bidaf(interact=True, streaming=False):
    global bidaf
    global datasource
    datasource = FileDialog("Bidaf", None)
    if datasource.start:
        settings['time_attribute'] = datasource.time
        settings['entity_attribute'] = datasource.entity
        settings['index_attribute'] = datasource.index
        settings['data_source_name'] = datasource.file_path.split("/")[-1]
        bidaf = BidafFramework(datasource, models, visualizers, settings)
        if streaming:
            bidaf.start()
        else:
            bidaf.handle_batch_data(datasource.all())
        if not interact:
            plt.ioff()
            plt.show()
            print("Done")
        else:
            return (bidaf, datasource)

if __name__ == "__main__":
    launch_bidaf(False, False)
