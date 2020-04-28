import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import datetime
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
import sys

""" TODO:
- Add possibility to color different segments of the time series
"""

class TimeSeriesVisualizer():

    def __init__(self, fig, rect, repo, callbacks):

        register_matplotlib_converters()
        plt.ion()
        
        # the preprocessing window slide step
        self.step = 10
        
        # show all entities or just selected ones
        self.show_all = True
        
        self.fig = fig
        # add 10% margin to rect [left, bottom, width, height]
        self.rect = [rect[0]+rect[2]*0.065, rect[1]+rect[3]*0.05, rect[2]*0.9, rect[3]*0.9]
        self.repo = repo
        self.callbacks = callbacks       
        
        if 'select_entities' not in callbacks:
            callbacks['select_entities'] = []
        if 'select_attributes' not in callbacks:
            callbacks['select_attributes'] = []
        self.callbacks['select_entities'].append(self.set_selected_entities)
        self.callbacks['select_attributes'].append(self.set_selected_attributes)
        self.draginfo = False

        self.indexattr = False
        self.timeattr = False
        self.entityattr = False
        self.classattr = False
        self.anomalyattr = False
        self.classcolfunc = None
        self.anomalycolfunc = None
        self.anomaly_threshold = 0.5
        
        self.all_entities = []
        self.selectedEntities = []
        self.selectedFeatures = []
        self.zoompropmin = 0.0
        self.zoompropmax = 1.0
        #self.timeWindowInHours = self.repo.get_time_window_seconds()/60/60

        # Create a list with rectangles for the plots
        rectangleList = self._getRectangleList(self.rect, len(self.selectedFeatures))

        # Create dict to hold all the feature information
        self.features = {}
        
        
        # Create anomalies dict to hold all the deviation plot info
        self.anomalies = { 'axes': fig.add_axes(rectangleList[0]),
                           'xMin': np.inf,#datetime.datetime.now(), 
                           'xMax': -np.inf,#datetime.datetime(1900, 1, 1),
                           'yMin': sys.maxsize, 
                           'yMax': -sys.maxsize - 1,
                           'entities': [] }
        
        self.anomalies['axes'].set_ylabel('')
        self.anomalies['axes'].set_title(self.anomalyattr if self.anomalyattr else "", fontdict={'fontSize': 10}, loc='left')
        self.anomalies['axes'].tick_params(labelsize=10)

            
    def add_entity(self, entity):
        
        # Only add entity when its not already in list
        if not entity in self.selectedEntities:
        
            # Add entity to list
            self.selectedEntities.append(entity)
            
            # Add entity to anomaly plot
            plotLine = self.anomalies['axes'].plot([], [])[0] #.scatter([], []) 
            plotLabel = 'Entity - ' + str(entity)
            
            # Append entity anomaly plot info to list
            self.anomalies['entities'].append({'entity': entity, 
                                               'label': plotLabel,
                                               'line': plotLine})


#        
#            # Add the entity to the legend (somehow the get_legend_handles_labels() did not return anything, 
#            # thats why the get_legend() workaround is used)
#            currentLegend = self.anomalies['axes'].get_legend()
#            legendLabels = [] if currentLegend is None else [str(x._text) for x in currentLegend.texts]
#            legendPlotLines = [] if currentLegend is None else currentLegend.legendHandles
#            
#            # Append the label and the plotline to the legend
#            legendPlotLines = np.append(legendPlotLines, plotLine)
#            legendLabels = np.append(legendLabels, plotLabel)
#            
#            # Set the legend for the entities / features
#            self.anomalies['axes'].legend(legendPlotLines, legendLabels)

            # Add entity to all feature plots
            for feature in self.features:
                plotInfo = { 'entity': entity,
                             'line': self.features[feature]['axes'].plot([], [])[0] } #.scatter([], [])}
                
                # Match the entity color in the anomaly plot
#                plotInfo['line'].set_color(plotLine.get_color())
                
                # Add entity plot info to feature entity list
                self.features[feature]['entities'].append(plotInfo)

                
    def remove_entity(self, entity):
        
        # Only if entity is in list
        if entity in self.selectedEntities:
        
            # Remove entity from list
            self.selectedEntities.remove(entity)

            # Remove entity entry from from all the feature plots
            for feature in self.features:
                
                for index, entityPlotInfo in enumerate(self.features[feature]['entities']):
                
                    if entityPlotInfo['entity'] == entity:
                        
                        # Remove line from plot
                        #self.features[feature]['axes'].lines.remove(entityPlotInfo['line'])
                        
                        # Remove list entry
                        self.features[feature]['entities'].pop(index)
                        
                        break

        
#      # Prepare to remove the entity from the legend (somehow the get_legend_handles_labels() did not return anything, 
#            # thats why the get_legend() workaround is used)
#            currentLegend = self.anomalies['axes'].get_legend()
#            legendLabels = [] if currentLegend is None else [str(x._text) for x in currentLegend.texts]
#            legendPlotLines = [] if currentLegend is None else currentLegend.legendHandles
#
            # Remove entity from anomaly plot
            for index, entityPlotInfo in enumerate(self.anomalies['entities']):
                
                if entityPlotInfo['entity'] == entity:

                    # Remove line from legend and plot (with workaround to remove plotline)
#                    legendPlotLines.pop(legendLabels.index(entityPlotInfo['label']))
#                    legendLabels.remove(entityPlotInfo['label'])

                    #self.anomalies['axes'].lines.remove(entityPlotInfo['line'])

                    # Remove list entry
                    self.anomalies['entities'].pop(index)
                                        
                    break

            # Set the legend for the entities / features
#            self.anomalies['axes'].legend(legendPlotLines, legendLabels)


    def add_feature(self, feature):
        
        # Only if feature is not already show
        if not feature in self.features:
        
            # Add the feature to the selected features
            self.selectedFeatures.append(feature)
        
            # Add a new feature dict in self.features, the None values will be replaced later
            self.features[feature] = { 'axes': None, 
                                       'feature': feature,
                                       'showTicks': None,
                                       'xMin': np.inf,#datetime.datetime.now(), 
                                       'xMax': -np.inf,#datetime.datetime(1900, 1, 1),
                                       'yMin': sys.maxsize, 
                                       'yMax': -sys.maxsize - 1,
                                       'entities': [] }

            # Add all the selected entities to the new feature, the None value will be replaced later
            for entity in self.selectedEntities:
                plotInfo = { 'entity': entity,
                             'line': None }
                self.features[feature]['entities'].append(plotInfo)

            # Get new rectangles for the feature plots
            rectangleList = self._getRectangleList(self.rect, len(self.selectedFeatures))
            
            # Replace axes for the features based on new rectangle layout
            for index, feature in enumerate(self.selectedFeatures):
                
                if self.features[feature]['axes'] != None:
                    self.features[feature]['axes'].remove()
                    
                self.features[feature]['axes'] = self.fig.add_axes(rectangleList[index + 1])
                self.features[feature]['showTicks'] = False #(index == 0)
                
                self.features[feature]['axes'].set_ylabel('')
                
                if not self.features[feature]['showTicks']:
                    self.features[feature]['axes'].set_xticklabels([])
                    
                self.features[feature]['axes'].set_title(feature, fontdict={'fontSize': 10}, loc='left')
                
                # Rebuild the entity list of the feature plot
                for entityIndex, entity in enumerate(self.selectedEntities):
                    plotInfo = { 'entity': entity,
                                 'line': self.features[feature]['axes'].plot([], [])[0] }  # .scatter([], [])}
                    
                    # Match the entity color in the anomalies plot
                    anomalyEntityPlotInfo = next(item for item in self.anomalies['entities'] if item['entity'] == entity)
#                    plotInfo['line'].set_color(anomalyEntityPlotInfo['line'].get_color())
                    
                    self.features[feature]['entities'][entityIndex] = plotInfo
                
                
                                
    def remove_feature(self, feature):
        
        # Only if feature is shown
        if feature in self.features:
            
            # Remove the feature 
            self.features[feature]['axes'].remove()
            del self.features[feature]
            self.selectedFeatures.remove(feature)
            
            # Get new rectangles for the features left
            rectangleList = self._getRectangleList(self.rect, len(self.selectedFeatures))
            
            # Replace axes for the still visible features based on new rectangle layout
            for index, feature in enumerate(self.selectedFeatures):
                
                self.features[feature]['axes'].remove()
                self.features[feature]['axes'] = self.fig.add_axes(rectangleList[index + 1])
                self.features[feature]['showTicks'] = False #(index == 0)
                
                self.features[feature]['axes'].set_ylabel('')
                
                if not self.features[feature]['showTicks']:
                    self.features[feature]['axes'].set_xticklabels([])
                    
                self.features[feature]['axes'].set_title(feature, fontdict={'fontSize': 10}, loc='left')
                
                # Rebuild the entity list of the feature plot
                for entityIndex, entity in enumerate(self.selectedEntities):
                    plotInfo = { 'entity': entity,
                                 'line': self.features[feature]['axes'].plot([], [])[0] } #.scatter([], [])}

                    # Match the entity color in the anomalies plot
                    anomalyEntityPlotInfo = next(item for item in self.anomalies['entities'] if item['entity'] == entity)
#                    plotInfo['line'].set_color(anomalyEntityPlotInfo['line'].get_color())

                    self.features[feature]['entities'][entityIndex] = plotInfo


                    
    def _get_cols(self, ano, cl):
        cols = [None]*len(cl)
        
        for i in range(len(cl)):
            
            if ano[i] is not None and ano[i] > self.anomaly_threshold and self.anomalycolfunc is not None:
                cols[i] = self.anomalycolfunc(ano[i])
            elif cl[i] is not None and self.classcolfunc is not None:
                cols[i] = self.classcolfunc(cl[i])
            else:
                cols[i] = hsl_color(0.5, 0.5, -0.5)
                
        return cols
    
    def _to_lines(self, points, colors):
        if len(points) != len(colors):
            print('Error!! Points and colors are not the same length')
            return
        
        x0, y0 = points[0]
        lines=[]
        new_colors=[]
        for i in range(1, len(points)):
            x1, y1 = points[i]
            #skip line if missing data in between
            if x1-x0 > self.step:
                x0,y0 = x1,y1
                continue
            lines.append( ((x0,y0) , (x1,y1)) )
            new_colors.append(colors[i])
            x0,y0 = x1,y1
        return lines, new_colors

    
    def redraw_features(self):
        
        # this will be a lits of lists in the form [[time, entity, anomaly, class, sel_feat_1, sel_feat_2, ...],...]
        #all_data = self.repo.aget_values([self.anomalyattr, self.classattr]+self.selectedFeatures, True)
        all_data = self.repo.current_data()
        ent_dat = {}
        for d in all_data:
            t = mdates.epoch2num(d[self.timeattr]) if self.timeattr is not False else d[self.indexattr]
            e = d[self.entityattr] if self.entityattr is not False else True
            if e not in self.all_entities:
                self.all_entities.append(e)
            v = [t, e] + [d[f] for f in [self.anomalyattr, self.classattr]+self.selectedFeatures]
            if e not in ent_dat.keys():
                ent_dat[e] = [v]
            else:
                ent_dat[e].append(v)
            
        # transpose for easy access to features
        for k in ent_dat:
            ent_dat[k] = list(zip(*ent_dat[k]))
        
        if self.show_all:
            if len(self.selectedEntities) < len(self.all_entities):
                self.set_selected_entities([])
        
        #print(all_data)
        #print(ent_dat)
                
        #all_data[0] = [ datetime.fromtimestamp(t) for t in all_data[0]]
        
        # for all selected features
        for f in enumerate(self.selectedFeatures,start=4): # start=4 because 0-time, 1-entity, 2-anomaly, 3-class, 4-features 
            # clear old data
            self.features[f[1]]['axes'].clear()
            self.features[f[1]]['axes'].set_title(f[1], fontdict={'fontSize': 10}, loc='left')
            self.features[f[1]]['axes'].set_ylabel('')
            self.features[f[1]]['axes'].set_xticklabels([])
            # and each entity in feature
            
            for entityPlotInfo in self.features[f[1]]['entities']:
                entity = entityPlotInfo['entity']
                dat = ent_dat[entity]
                if(len(dat) == 0):
                    continue
                
                p = list(zip(dat[0], dat[f[0]]))
                c = self._get_cols(dat[2],dat[3])
                
                l,cc = self._to_lines(p,c)
                colored_lines = LineCollection(l, colors=cc, linewidths=(3,))
                self.features[f[1]]['axes'].add_collection(colored_lines)
                #entityPlotInfo['line'].set_offsets( p  )
                #entityPlotInfo['line'].set_color(c)
                
                self._reAdjustMultiPlotLimits(self.features[f[1]], min(dat[0]), min(dat[f[0]]))
                self._reAdjustMultiPlotLimits(self.features[f[1]], max(dat[0]), max(dat[f[0]]))
        
        #Update the deviation plot
        self.anomalies['axes'].clear()
        self.anomalies['axes'].set_ylabel('')
        self.anomalies['axes'].set_title(self.anomalyattr if self.anomalyattr else "", fontdict={'fontSize': 10}, loc='left')
        
        for entityPlotInfo in self.anomalies['entities']:
            entity = entityPlotInfo['entity']
            dat = ent_dat[entity]
            
            
            if len(dat) > 0:

                p = list(zip(dat[0], dat[2]))
                c = self._get_cols(dat[2], dat[3])
                l,cc = self._to_lines(p,c)
                colored_lines = LineCollection(l, colors=cc, linewidths=(3,))
                self.anomalies['axes'].add_collection(colored_lines)
                
                #entityPlotInfo['line'].set_offsets( list(zip(dat[0], dat[2])) )
                #entityPlotInfo['line'].set_color(self._get_cols(dat[3]))
                
                dd = [d for d in dat[2] if d is not None]
                if dd:
                    self._reAdjustMultiPlotLimits(self.anomalies, min(dat[0]), min(dd))
                    self._reAdjustMultiPlotLimits(self.anomalies, max(dat[0]), max(dd))
                
        if len(dat)>0 and self.timeattr is not False:
            ax = self.anomalies['axes']
            #ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
            #ax.xaxis.set_minor_locator(mdates.MinuteLocator())
            monthFmt = mdates.DateFormatter("%H:%M")
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(monthFmt)
    
    
    def handle_data(self, dict_msg):
        
        entity = (dict_msg[self.entityattr] if self.entityattr is not False else True)
        if not entity in self.all_entities:
            self.all_entities.append(entity)

        # Only act if the entity is in the entity list
        if entity in self.selectedEntities:

            if self.timeattr is not False:
                newXvalue = mdates.epoch2num(dict_msg[self.timeattr])
                #newXvalue = datetime.datetime.fromtimestamp(dict_msg[self.timeattr])
            else:
                newXvalue = dict_msg[self.indexattr]
                
            # For every selected feature
            for feature in self.features:

                newYvalue = dict_msg[feature]
                featurePlotInfo = self.features[feature]
                
                # Get the entity information from the feature plot
                entityPlotInfo = next(item for item in featurePlotInfo['entities'] if item["entity"] == entity)

                newXvalues = np.append(entityPlotInfo['line'].get_xdata(), newXvalue)
                newYvalues = np.append(entityPlotInfo['line'].get_ydata(), newYvalue)

                self._set_data(entityPlotInfo['line'], newXvalues, newYvalues)
                self._reAdjustMultiPlotLimits(featurePlotInfo, newXvalue, newYvalue)
                self._removeDataNotVisible(featurePlotInfo['xMin'], entityPlotInfo['line'])
                    
            # Update the deviation plot of the entity 
            if self.anomalyattr is not False and dict_msg[self.anomalyattr] is not None:
                
                anomalyPlotInfo = next(item for item in self.anomalies['entities'] if item['entity'] == entity)
                
                xx = np.append(anomalyPlotInfo['line'].get_xdata(), newXvalue)
                yy = np.append(anomalyPlotInfo['line'].get_ydata(), dict_msg[self.anomalyattr])
                
                self._set_data(anomalyPlotInfo['line'], xx, yy)
                self._reAdjustMultiPlotLimits(self.anomalies, 
                                              newXvalue, 
                                              dict_msg[self.anomalyattr])
                self._removeDataNotVisible(self.anomalies['xMin'], anomalyPlotInfo['line'])
                self.anomalies['axes'].set_ylim(0, 1)
                
    
    def _set_data(self, line, xx, yy):
        
        line.set_xdata(xx)
        line.set_ydata(yy)

        
    def _removeDataNotVisible(self, xMin, line):
        pass
        # Remove the data that is not shown to improve memory use
        #line.set_xdata([x for x in line.get_xdata() if x >= xMin])
        #line.set_ydata(line.get_ydata()[-len(line.get_xdata()):])
        
        
    def _reAdjustMultiPlotLimits(self, plotInfo, x, y):
        if x is not None and y is not None:
            # Determine the new min and max values
            if x < plotInfo['xMin']:
                plotInfo['xMin'] = x

            if x > plotInfo['xMax']:
                plotInfo['xMax'] = x

            if y < plotInfo['yMin']:
                plotInfo['yMin'] = y

            if y > plotInfo['yMax']:
                plotInfo['yMax'] = y

        # Readjusting the x axis according the requested time window
        #xMax = datetime.fromtimestamp(self.repo.get_time_now())#plotInfo['xMax']
        #xMin = xMax - timedelta(hours = self.timeWindowInHours)
        xMax = plotInfo['xMax']
        xMin = plotInfo['xMin']
        
        # Set the x axis limits
        if xMin != xMax:
            plotInfo['axes'].set_xlim(xMin + (xMax-xMin)*self.zoompropmin, xMin + (xMax-xMin)*self.zoompropmax)
               
        # Adjust the y min and y max
        ydiff = plotInfo['yMax'] - plotInfo['yMin']
        if (ydiff > 0.0):
            yMin = plotInfo['yMin'] - 0.05 * ydiff
            yMax = plotInfo['yMax'] + 0.05 * ydiff
        else:
            yMin = plotInfo['yMin'] - 0.1
            yMax = plotInfo['yMax'] + 0.1
            
        plotInfo['axes'].set_ylim(yMin, yMax)
        
        
    def _getRectangleList(self, rect, numberOfFeatures):
        
        # Holds the list with rectangles to be returned
        rectangles = []
        
        # Just for easier reading
        margin = 0.01
        areaLeft = rect[0]
        areaBottom = rect[1]
        areaWidth = rect[2]
        areaHeight = rect[3]

        # The deviation rectangle takes half the space on the bottom
        rectangles.append([areaLeft, areaBottom, areaWidth, areaHeight / 4 - margin])
        
        if numberOfFeatures > 0:
            
            # The top half of the space is split among the feature rectangles
            featureRectangleHeight = (areaHeight*3 / 4 / numberOfFeatures) - margin

            previousBottom = 0

            for featureOrder in range(0, numberOfFeatures):

                if featureOrder == 0:
                    featureBottom = (areaBottom + (areaHeight / 4)) + (1.5 * margin)
                else:
                    featureBottom = previousBottom + featureRectangleHeight + (2.5 * margin)

                rectangles.append([areaLeft, featureBottom, areaWidth, featureRectangleHeight])

                previousBottom = featureBottom

        return rectangles

    
    def set_selected_entities(self,entity_list):
        if not entity_list:
            #the list is empty which means all entities
            entity_list = self.all_entities
            self.show_all = True
        else:
            self.show_all = False
            
        toadd = [x for x in entity_list if x not in self.selectedEntities]
        toremove = [x for x in self.selectedEntities if x not in entity_list]
        for i in toadd:
            self.add_entity(i)
        for i in toremove:
            self.remove_entity(i)
        self.redraw_features()
                           
    def set_selected_attributes(self,attribute_list):
        features_to_show = 3
        
        if attribute_list:
            toadd = [x for x in attribute_list if x not in self.selectedFeatures]
            toremove = [x for x in self.selectedFeatures if x not in attribute_list]
            n_remove = len(toadd)+len(self.selectedFeatures)-features_to_show
            for i in toadd:
                self.add_feature(i)
            for i in range(0,n_remove):
                self.remove_feature(toremove[i])
        self.redraw_features()

    
    def default_params(self):
        return { 'index_attribute': False,
                 'time_attribute': False,
                 'entity_attribute': False,
                 'class_attribute': False,
                 'class_color': None,
                 'anomaly_attribute': False, 
                 'anomaly_color': None,
                 'anomaly_threshold': 0.5 }

    def set_params(self, dic):
        if 'time_attribute' in dic:
            self.timeattr = dic['time_attribute']
        if 'index_attribute' in dic:
            self.indexattr = dic['index_attribute']
        if 'entity_attribute' in dic:
            self.entityattr = dic['entity_attribute']
        if 'class_attribute' in dic:
            self.classattr = dic['class_attribute']
        if 'class_color' in dic:
            self.classcolfunc = dic['class_color']
        if 'anomaly_attribute' in dic:
            self.anomalyattr = dic['anomaly_attribute']
        if 'anomaly_color' in dic:
            self.anomalycolfunc = dic['anomaly_color']
        if 'anomaly_threshold' in dic:
            self.anomaly_threshold = dic['anomaly_threshold']

    def redraw_model(self, moddict):
        pass
    
    def scroll_event(self, event):
        # p, pmin, pmax, fact -> qmin, qmax
        # (qmx-qmin) = (pmax-pmin)*fact
        # qmin + p*(qmax-qmin) = pmin + p*(pmax-pmin)
        ax = self.anomalies['axes']
        tr = ax.transAxes.inverted().transform((event.x,event.y))
        pp = tr[0]
        if event.button == "up":
            fact = 0.8
        else:
            fact = 1.25
        pdiff = (self.zoompropmax - self.zoompropmin)
        qdiff = min(1.0, (self.zoompropmax - self.zoompropmin)*fact)
        self.zoompropmin = min(1.0, max(0.0, self.zoompropmin + pp*(pdiff-qdiff)))
        self.zoompropmax = max(0.0, min(1.0, self.zoompropmin + qdiff))
        for feature in self.features:
            self._reAdjustMultiPlotLimits(self.features[feature], None, None)
        self._reAdjustMultiPlotLimits(self.anomalies, None, None)
        plt.draw()

    def button_press_event(self, event):
        if event.button == 1 and event.key == None:
            ax = self.anomalies['axes']
            trans = ax.transAxes.inverted()
            pp = trans.transform((event.x,event.y))[0]
            if pp >= 0.0 and pp <= 1.0:
                self.draginfo = (trans, pp, self.zoompropmin, self.zoompropmax)
            else:
                self.draginfo = False

    def motion_notify_event(self, event):
        if self.draginfo is not False:
            (trans, oldpp, oldzmin, oldzmax) = self.draginfo
            pp = max(0.0, min(1.0, trans.transform((event.x,event.y))[0]))
            diff = -(pp-oldpp)*(oldzmax-oldzmin)
            self.zoompropmin = min(1.0-(oldzmax-oldzmin), max(0.0, oldzmin + diff))
            self.zoompropmax = max(0.0+(oldzmax-oldzmin), min(1.0, oldzmax + diff))
            for feature in self.features:
                self._reAdjustMultiPlotLimits(self.features[feature], None, None)
            self._reAdjustMultiPlotLimits(self.anomalies, None, None)
            plt.draw()

    def button_release_event(self, event):
        if event.button == 1 and self.draginfo is not False:
            self.draginfo = False
