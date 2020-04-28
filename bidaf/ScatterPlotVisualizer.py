import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from random import *
from sklearn.decomposition import PCA

from Color import *

def cluster_color(ind):
    gs = 1.5-np.sqrt(1.25)
    return hsl_color((gs*ind)%1.0, 1.0, 0.0)

def anomaly_color(ano):
    if ano == 0.0:
        return False
    elif ano > 1.0:
        return hsl_color(0.0, 0.0, -1.0)
    else:
        return hsl_color(0.0, 0.0, 0.4-ano*1.4)

def varians_to_ellipse(var, a1, a2):
    s1=var[a1][a1]
    s2=var[a2][a2]
    s12=var[a1][a2]
    # (s1-a  s12 ) * (v1) = ( (s1-a)*v1 + s12*v2 ) = 0
    # (s12   s2-a)   (v2)   ( s12*v1 + (s2-a)*v2 )
    # 0 = (s1-a)*v1 + s12*v2  ->  (a-s1)/s21 = v2/v1
    # 0 = (s1-a)(s2-a) - s12^2 = (s1 s2 - s12^2) -(s1+s2)a + a^2
    # a = +-sqrt((s1-s2)^2 / 4 + s12^2) + (s1+s2)/2
    sq = np.sqrt((s1-s2)*(s1-s2)/4 + s12*s12)
    a1 = (s1+s2)/2 + sq
    a2 = (s1+s2)/2 - sq
    v = np.arctan2((a1 - s1), s12)
    return (np.sqrt(a1), np.sqrt(a2), v*180.0/np.pi)

def gaussoutline(mean, var, a1, a2, dist, col='black'):
    cent = (mean[a1], mean[a2])
    ell = varians_to_ellipse(var, a1, a2)
    return mpl.patches.Ellipse(cent, ell[0]*2*dist, ell[1]*2*dist, angle=ell[2], edgecolor=col, fill=False)


class ScatterPlotVisualizer():
    def __init__(self, fig, rect, repo, callbacks):
        self.callbacks = callbacks
        self.repo = repo

        # 'select_entity/ies', 'select_attribute/s', 'select_timespan'
        if not 'select_attributes' in self.callbacks:
            self.callbacks['select_attributes'] = []
        if not 'select_entities' in self.callbacks:
            self.callbacks['select_entities'] = []
        self.callbacks['select_entities'].append(self.set_selected_entities)
        self.callbacks['select_attributes'].append(self.set_selected_attributes)
        self.selected_entities = []
        self.selected_attributes = []
        self.select_parity = 0
        self.classattr = False
        self.anomalyattr = False
        self.classcolfunc = None
        self.anomalycolfunc = None
        self.anomaly_threshold = 0.5

        self.fig = fig
        self.offx = rect[0]
        self.offy = rect[1]
        inches = fig.get_size_inches()
        self.width = int(fig.dpi*inches[0]*rect[2])
        self.height = int(fig.dpi*inches[1]*rect[3])
        inner = (rect[0]+rect[2]/16, rect[1]+rect[3]/16, rect[2]*0.90625, rect[3]*0.90625)
        self.ax = fig.add_axes(inner)
        self.sc = self.ax.scatter([], [])
        self.sc.set_sizes([20])
        self.sc.set_zorder(1.0)
        self.sc0 = self.ax.scatter([], [])
        self.sc0.set_sizes([20])
        self.sc0.set_zorder(0.5)
        self.model = []
        self.outlines = []
        self.pca = False

    def handle_data(self, vecdic):
        #self.draw_data()
        #self.draw_model()
        return None

    def redraw_model(self, moddic):
        if moddic['type'] == 'ClusterModel':
            self.model = moddic['mixture']
            self.draw_model()

    def redraw_features(self):
        self.draw_data()
        self.draw_model()

    def default_params(self):
        return { 'entity_attribute': False,
                 'class_attribute': False,
                 'class_color': None,
                 'anomaly_attribute': False, 
                 'anomaly_color': None,
                 'anomaly_threshold': 0.5 }
    
    def set_params(self, dic):
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


    #------------------------------

    def draw_data(self):
        # om ingen selectad, pca
        # om två selectade, plotta
        # om en selectad, använd en från senaste...
        # ljusgrå om ej selectad entity
        # klustrets färg, eller anomalins färg, eller avtar med tiden
        #keys = self.repo.get_keys()
        feat = self.repo.get_feature_names()
        dictlst = self.repo.current_data()
        #values = self.repo.get_values(feat)
        #status = self.repo.get_values([self.anomalyattr,self.classattr])
        if len(dictlst) < 3: # **** or len(feat) < 2:
            return
        if len(self.selected_attributes) < 2:
            # prepare pca
            values = list(map(lambda v: tuple([v[f] for f in feat]), dictlst))
            self.pca = PCA(n_components = 2).fit(values) 
            data = self.pca.transform(values)
            self.ax.set_xlabel("PCA 1")
            self.ax.set_ylabel("PCA 2")
        else:
            ind1 = self.selected_attributes[0]
            ind2 = self.selected_attributes[1]
            data = list(map(lambda v: (v[ind1], v[ind2]), dictlst))
            self.ax.set_xlabel(self.selected_attributes[0])
            self.ax.set_ylabel(self.selected_attributes[1])
        min1 = np.inf
        min2 = np.inf
        max1 = -np.inf
        max2 = -np.inf
        for vec in data:
            if vec[0] < min1: min1 = vec[0]
            if vec[0] > max1: max1 = vec[0]
            if vec[1] < min2: min2 = vec[1]
            if vec[1] > max2: max2 = vec[1]
        if not (np.isinf(min1) or np.isinf(min2) or np.isinf(max1) or np.isinf(max2)):
            delta1 = (max1-min1)/16
            delta2 = (max2-min2)/16
            self.ax.set_xlim((min1-delta1, max1+delta1))
            self.ax.set_ylim((min2-delta2, max2+delta2))
        cols = [None]*len(data)
        g_ind = []
        #for (i, (t,e)) in enumerate(keys):
        for (i, dic) in enumerate(dictlst):
            if len(self.selected_entities) == 0 or self.entityattr is False or dic[self.entityattr] in self.selected_entities:
                if dictlst[i][self.anomalyattr] is not None and dictlst[i][self.anomalyattr] > self.anomaly_threshold and self.anomalycolfunc is not None:
                    cols[i] = self.anomalycolfunc(dictlst[i][self.anomalyattr])
                elif dictlst[i][self.classattr] is not None and self.classcolfunc is not None:
                    cols[i] = self.classcolfunc(dictlst[i][self.classattr])
                else:
                    cols[i] = hsl_color(0.5, 0.5, -0.5)
            else:
                cols[i] = hsl_color(0.0, 0.0, 0.75)
                g_ind.append(i)
                
        self.sc0.set_offsets([x for i,x in enumerate(data) if i in g_ind])
        self.sc0.set_color([x for i,x in enumerate(cols) if i in g_ind])
        self.sc.set_offsets([x for i,x in enumerate(data) if i not in g_ind])
        self.sc.set_color([x for i,x in enumerate(cols) if i not in g_ind])
        #self.sc.set_zorder(zcoord)

    def draw_model(self):
        for o in self.outlines:
            o.remove()
        self.outlines = []
        if len(self.selected_attributes) < 2:
            if self.pca is not False:
                for g in self.model:
                    mean = np.matmul(self.pca.components_, np.array(g['mean'])-self.pca.mean_)
                    var = np.matmul(self.pca.components_, np.matmul(np.array(g['var']), np.transpose(self.pca.components_)))
                    o = gaussoutline(mean, var, 0, 1, 1.0)
                    self.ax.add_artist(o)
                    self.outlines.append(o)
        else:
            feat = self.repo.get_feature_names()
            ind1 = feat.index(self.selected_attributes[0])
            ind2 = feat.index(self.selected_attributes[1])
            for g in self.model:
                o = gaussoutline(g['mean'], g['var'], ind1, ind2, 1.0)
                self.ax.add_artist(o)
                self.outlines.append(o)
        
    def set_selected_entities(self,entity_list):
        self.selected_entities = entity_list
        self.draw_data()

    def set_selected_attributes(self,attribute_list):
        if not attribute_list:
            self.selected_attributes = []
            self.select_parity = 0
        elif len(attribute_list) == 1:
            if not self.selected_attributes:
                self.selected_attributes = [attribute_list[0]]
                self.select_parity = 1
            elif len(self.selected_attributes) == 1:
                self.selected_attributes.append(attribute_list[0])
                self.select_parity = 0
            else:
                self.selected_attributes[self.select_parity] = attribute_list[0]
                self.select_parity = 1 - self.select_parity
        else:
            self.selected_attributes = [attribute_list[0],attribute_list[1]]
            self.select_parity = 0
        self.draw_data()
        self.draw_model()

    #def trans(self, pos):
    #    tr = self.fig.transFigure.inverted().transform(pos)
    #    return (tr[0]+self.offx, tr[1]+self.offy)

    #def xscale(self):
    #    return self.trans((1,1))[0] - self.trans((0,0))[0]

    #def yscale(self):
    #    return self.trans((1,1))[1] - self.trans((0,0))[1]

