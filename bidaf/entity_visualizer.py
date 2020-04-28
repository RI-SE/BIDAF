import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from math import *

from Color import *

from graph_visualizer import GraphMessage


def tupleadd(x, y, mul=1):
    return (x[0]+mul*y[0], x[1]+mul*y[1])

    
class Entity():
    def __init__(self, vis, id, pos, sz, sel = False):
        self.vis = vis
        self.id = id
        self.name = str(id)
        self.pos = pos
        self.sqsize = sz
        self.selected = sel
        self.color = 'white'
        xsc = vis.xscale()
        ysc = vis.yscale()
        p1 = vis.trans(self.pos)
        p2 = tupleadd(p1, (xsc, -ysc), 2.0)
        p3 = tupleadd(p1, (xsc, -ysc), -1.0)
        gr1 = hsl_color(0, 0, -0.2)
        gr2 = hsl_color(0, 0, -0.5)
        gr3 = hsl_color(0, 0, 0.3)
        self.txt = plt.Text(p1[0] + (self.sqsize + 10)*xsc, p1[1] + self.sqsize*ysc/3, self.name, ha='left', fontsize=self.vis.fontsz*self.vis.dotwidth)
        r1 = plt.Rectangle(tupleadd(p1, (xsc, -ysc), 1.0), xsc*self.sqsize, ysc*self.sqsize, linewidth=6*self.vis.dotwidth, edgecolor= gr1, facecolor=self.color)
        r2 = plt.Rectangle(p2, xsc*self.sqsize, ysc*self.sqsize, linewidth=3*self.vis.dotwidth, edgecolor= gr2, fill=False)
        r3 = plt.Rectangle(p3, xsc*self.sqsize, ysc*self.sqsize, linewidth=3*self.vis.dotwidth, edgecolor= gr3, fill=False)
        if self.selected:
            r4 = plt.Rectangle(tupleadd(p1, (0, -ysc), 2.0), xsc*(self.sqsize+2), ysc*(self.sqsize+2), linewidth=2*self.vis.dotwidth, edgecolor= gr1, fill=False)
        else:
            r4 = plt.Rectangle(tupleadd(p1, (xsc, 0), 2.0), xsc*(self.sqsize-2), ysc*(self.sqsize-2), linewidth=2*self.vis.dotwidth, edgecolor= gr1, fill=False)
        self.objs = [r1,r2,r3,r4]
        for r in self.objs:
            self.vis.fig.add_artist(r)
        self.vis.fig.add_artist(self.txt)

    def set_color(self, col):
        self.color = col
        self.objs[0].set_facecolor(col)

    def set_text_color(self, col):
        self.txt.set_color(col)

    def set_selected(self, val):
        self.selected = val
        xsc = self.vis.xscale()
        ysc = self.vis.yscale()
        p1 = self.vis.trans(self.pos)
        if val:
            self.objs[3].set_bounds(p1[0], p1[1]-ysc*2, xsc*(self.sqsize+2), ysc*(self.sqsize+2))
        else:
            self.objs[3].set_bounds(p1[0]+xsc*2, p1[1], xsc*(self.sqsize-2), ysc*(self.sqsize-2))

    def set_position(self, pos):
        diff = tupleadd(self.vis.trans(pos), self.vis.trans(self.pos), -1)
        self.pos = pos
        for obj in self.objs:
            obj.set_xy(tupleadd(obj.get_xy(), diff, 1))
        self.txt.set_x(self.txt._x + diff[0])
        self.txt.set_y(self.txt._y + diff[1])


class EntityVisualizer():
    def __init__(self, fig, rect, repo, callbacks):
        self.callbacks = callbacks
        self.repo = repo
        # 'select_entity/ies', 'select_attribute/s', 'select_timespan'
        if not 'select_entities' in self.callbacks:
            self.callbacks['select_entities'] = []
        self.fig = fig
        self.offx = rect[0]
        self.offy = rect[1]
        inches = fig.get_size_inches()
        self.width = int(fig.dpi*inches[0]*rect[2])
        self.height = int(fig.dpi*inches[1]*rect[3])
        self.fontsz = 18
        self.sqsize = 36
        self.sqdist = int(self.sqsize/2)
        self.dotwidth = 0.75
        self.entities = {}
        self.selected = []
        #self.lasttimes = {}
        self.pressedentity = None
        self.pressedshift = False
        self.entityattr = False
        self.classattr = False
        self.anomalyattr = False
        self.classcolfunc = None
        self.anomalycolfunc = None
        self.count = 0
        self.message_count = GraphMessage(self, (self.width/2, self.height-self.sqsize/2), "Data count:")
        self.message_name = GraphMessage(self, (self.sqsize, self.height-self.sqsize/2), "")

    def handle_data(self, vecdic):
        # Check for new entities and add them
        # Maybe update color of existing entities
        # tm = vecdic['time']
        if self.entityattr is not False:
            id = vecdic[self.entityattr]
            cls = vecdic[self.classattr] if self.classattr in vecdic else None
            ano = vecdic[self.anomalyattr] if self.anomalyattr in vecdic else None
            #self.lasttimes[id] = tm
            if not id in self.entities:
                self.entities[id] = Entity(self, id, (0,0), self.sqsize)
                if not self.selected == []:
                    self.entities[id].set_text_color(hsl_color(0, 0, -0.3))                
                self.arrange_entities()
        #if ano is not None and ano > self.anomaly_threshold and self.anomalycolfunc is not None:
        #    self.entities[id].set_color(self.anomalycolfunc(ano))
        #elif cls is not None and cls is not False and self.classcolfunc is not None:
        #    self.entities[id].set_color(self.classcolfunc(cls))
        self.count += 1
        self.message_count.set_message("Data count: " + str(self.count), 0)
        return None

    def redraw_model(self, moddic):
        return None

    def redraw_features(self):
        if self.count == 0:
            if self.entityattr is not False:
                for vecdic in self.repo.current_data():
                    id = vecdic[self.entityattr]
                    if not id in self.entities:
                        self.entities[id] = Entity(self, id, (0,0), self.sqsize)
                self.arrange_entities()
            self.count = self.repo.len()
            self.message_count.set_message("Data count: " + str(self.count), 0)
        dictlst = self.repo.current_data()
        for id in self.entities:
            dictlst = self.repo.current_data()
            #classes = self.repo.get(id, self.classattr)
            #anomalies = self.repo.get(id, self.anomalyattr)
            i=len(dictlst)-1
            while i>=0:
                if dictlst[i][self.entityattr] == id and dictlst[i][self.classattr] is not None:
                    break
                i -= 1
            cls = dictlst[i][self.classattr] if i>0 else None
            i=len(dictlst)-1
            while i>=0:
                if dictlst[i][self.entityattr] == id and dictlst[i][self.anomalyattr] is not None:
                    break
                i -= 1
            ano = dictlst[i][self.anomalyattr] if i>0 else None
            if ano is not None and ano > self.anomaly_threshold and self.anomalycolfunc is not None:
                self.entities[id].set_color(self.anomalycolfunc(ano))
            elif cls is not None and cls is not False and self.classcolfunc is not None:
                self.entities[id].set_color(self.classcolfunc(cls))
        return None

        #for vecdic in self.repo:
        #    tm = vecdic['time']
        #    id = vecdic['entity']
        #    if id in lasttimes and lasttimes[id] == tm:
        #        cls = vecdic[self.classattr] if self.classattr in vecdic else None
        #        ano = vecdic[self.anomalattr] if self.anomalyattr in vecdic else None
        #        if ano is not None and ano > 0.0 and self.anomalycolfunc is not None:
        #            self.entities[id].set_color(self.anomalycolfunc(ano))
        #        elif cls is not None and cls is not False and self.classcolfunc is not None:
        #            self.entities[id].set_color(self.classcolfunc(cls))

    def default_params(self):
        return { 'entity_attribute': False,
                 'class_attribute': False,
                 'class_color': None,
                 'anomaly_attribute': False, 
                 'anomaly_color': None,
                 'anomaly_threshold': None,
                 'data_source_name':''}

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
        if 'data_source_name' in dic:
            self.message_name.set_message(dic['data_source_name'], 0)


    def button_press_event(self, event):
        #print("Press", event.button)
        ent = self.locate_entity(event)
        if ent is not False:
            self.pressedentity = ent
            self.pressedshift = event.key == 'shift'
            ent.set_selected(True)
            plt.draw()

    def button_release_event(self, event):
        #print("Release", event.button)
        if self.pressedentity is not None:
            gr = hsl_color(0, 0, -0.3)
            ent = self.pressedentity
            if self.pressedshift:
                if ent.id in self.selected:
                    self.selected.remove(ent.id)
                    ent.set_selected(False)
                else:
                    self.selected.append(ent.id)
            else:
                ok = not ent.id in self.selected
                for id in self.selected:
                    self.entities[id].set_selected(False)
                if ok:
                    self.selected = [ent.id]
                else:
                    self.selected = []
            self.pressedentity = None
            self.pressedshift = False
            for id in self.entities:
                self.entities[id].set_text_color('black' if self.selected==[] or id in self.selected else gr)
            for func in self.callbacks['select_entities']:
                func(self.selected)
            plt.draw()

    #------------------------------

    def trans(self, pos):
        tr = self.fig.transFigure.inverted().transform(pos)
        return (tr[0]+self.offx, tr[1]+self.offy)

    def xscale(self):
        return self.trans((1,1))[0] - self.trans((0,0))[0]

    def yscale(self):
        return self.trans((1,1))[1] - self.trans((0,0))[1]

    def locate_entity(self, event):
        for id in self.entities:
            if self.entities[id].objs[0].contains(event)[0]:
                return self.entities[id]
        return False

    def arrange_entities(self):
        num = len(self.entities)
        nrows = floor((self.height - self.sqdist)/(self.sqsize + self.sqdist))
        ncols = ceil(num/nrows)
        colwidth = ceil((self.width - self.sqdist*2)/ncols)
        yoff = self.height - self.sqsize - floor((self.height + self.sqdist - nrows * (self.sqsize + self.sqdist)) / 2)
        entlst = sorted(list(self.entities.keys()))
        for x in range(ncols):
            for y in range(nrows):
                ind = x*nrows + y
                if ind >= num:
                    return
                self.entities[entlst[ind]].set_position((self.sqdist + x*colwidth, yoff - y*(self.sqsize + self.sqdist)))

