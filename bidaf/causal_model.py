from math import * 
import re
import time

from citest import *  # make_format, signif, trisignif, distr, hist_maxinterval



def deepmem(ele, lst):
    if ele == lst:
        return True
    elif not type(lst) is list or lst==[]:
        return False
    else:
        return deepmem(ele, lst[0]) or deepmem(ele,lst[1:])

def subsets(lst, ord):
    if ord==0:
        return [[]]
    elif  ord > len(lst):
        return []
    else:
        rest = subsets(lst[1:], ord-1)
        return list(map(lambda l:[lst[0]]+l, rest)) + subsets(lst[1:],ord)

def find_node_sets(l1, l2, lev):
    s1 = subsets(l1, lev)
    s2 = subsets(l2, lev)
    for s in s2:
        if not s in s1:
            s1 += [s]
    return s1

def find_node_sets3(l1, l2, l3, lev):
    s1 = subsets(l1, lev)
    s2 = subsets(l2, lev)
    s3 = subsets(l3, lev)
    for s in s2 + s3:
        if not s in s1:
            s1 += [s]
    return s1

def saferemove(lst, ele):
    while ele in lst:
        lst.remove(ele)
    return lst

def saferemove2(lst, eles):
    for ele in eles:
        while ele in lst:
            lst.remove(ele)
    return lst

def find_node_index(nodes, name):
    ind = 0
    for n in nodes:
        if n.name == name:
            return ind
        ind += 1

def find_linked_nodes(links, name):
    res = []
    for e in links:
        if (e.status in ['Causal', 'Direct']):
            if e.x1 == name:
                res += [e.x2]
            elif e.x2 == name:
                res += [e.x1]
    return res

def find_link(links, nm1, nm2):
    for e in links:
        if (e.x1 == nm1 and e.x2 == nm2) or (e.x1 == nm2 and e.x2 == nm1):
            return e
    return None


class Node:
    def __init__(self, gr, name):
        self.name = name
        self.graph = gr
        self.prob = False
        self.hist = False

    def update_prob(self):
        (pr, h) = distr(self.graph.data, find_node_index(self.graph.nodes, self.name),
                        self.graph.citest, self.graph.form)
        self.prob = pr
        self.hist = h


class Edge:
    def __init__(self, gr, x1, x2):
        self.graph =gr
        self.x1 = x1
        self.x2 = x2
        self.reset()

    def reset(self):
        self.indir = (0.0, 0.0, 0.0, 1.0)
        self.dir = (0.0, 0.0, 0.0, 1.0)
        self.ndcond = []
        self.status = 'Unconfirmed'
        self.oldstatus = 'Unconfirmed'
        self.hist = False
        self.changed = False

    def update_indirect(self):
        sg, mean, h = signif(self.graph.data, [find_node_index(self.graph.nodes, self.x1),
                                          find_node_index(self.graph.nodes, self.x2)],
                             [], self.graph.citest, self.graph.form, self.graph.tolerance)
        mn, mx = hist_maxinterval(h, 1.0 - self.graph.significance*2) if h is not False else (mean, mean)
        self.indir = (mean, mn, mx, sg) # safesqrt(var), 
        self.hist = h
        newstat = 'Indirect' if sg < self.graph.significance else 'Unconfirmed'
        if newstat != self.status:
            self.status = newstat
            self.changed = True
            #if not silent:
            print("1: Making [" + self.x1 + "," + self.x2 + "] " + newstat)

    def update_direct(self, lev):
        # hitta alla grann-mängder av ordning lev
        # om inga -> direct
        # om betingning ger 0 -> not-direct
        if self.status == 'Unconfirmed' or self.status == 'Indirect':
            return
        cs = find_node_sets(saferemove(find_linked_nodes(self.graph.links, self.x1), self.x2),
                            saferemove(find_linked_nodes(self.graph.links, self.x2), self.x1), lev)
        for cond in cs:
            sg, mean, h = signif(self.graph.data,
                                 list(map(lambda n:find_node_index(self.graph.nodes, n), [self.x1, self.x2])),
                                 list(map(lambda n:find_node_index(self.graph.nodes, n), cond)),
                                 self.graph.citest, self.graph.form, self.graph.tolerance)
            if sg > self.graph.significance:
                self.ndcond += [cond]
                if self.status == 'Direct':
                    self.status = 'Indirect'
                #if not silent:
                print("2: Confirming [" + self.x1 + "," + self.x2 + "] " + 'Non-direct')


class HyperEdge:
    def __init__(self, x1, x2, x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.reset()

    def reset(self):
        self.indir = (0.0, 0.0, 0.0, 1.0)
        self.dir = (0.0, 0.0, 0.0, 1.0)
        self.ndcond = []
        self.status = 'Unconfirmed'
        self.oldstatus = 'Unconfirmed'
        self.hist = False
        self.changed = False

    def update_indirect(self):
        sg, mean, h = trisignif(self.graph.data,
                                list(map(lambda n:find_node_index(self.graph.nodes, n), [self.x1, self.x2, self.x3])),
                                [], self.graph.citest, self.graph.form, self.graph.tolerance)
        mn, mx = hist_maxinterval(h, 1.0 - self.graph.significance*2) if h is not False else (mean, mean)
        self.indir = (mean, mn, mx, sg)
        self.hist = h
        newstat = 'Indirect' if sg < self.graph.significance else 'Unconfirmed'
        if newstat != self.status:
            self.status = newstat
            self.changed = True
            #if not silent:
            print("H1: Making [" + self.x1 + "," + self.x2 + "," + self.x3 + "] " + newstat)

    def update_direct(self, lev):
        if self.status == 'Unconfirmed' or self.status == 'Indirect':
            return
        cs = find_node_sets3(saferemove2(find_linked_nodes(self.graph.links, self.x1), [self.x2, self.x3]),
                             saferemove2(find_linked_nodes(self.graph.links, self.x2), [self.x1, self.x3]),
                             saferemove2(find_linked_nodes(self.graph.links, self.x3), [self.x1, self.x2]), lev)
        for cond in cs:
            sg, mean, h = trisignif(self.graph.data,
                                    list(map(lambda n:find_node_index(self.graph.nodes, n), [self.x1, self.x2, self.x3])),
                                    list(map(lambda n:find_node_index(self.graph.nodes, n), cond)),
                                    self.graph.citest, self.graph.form, self.graph.tolerance)
            mn, mx = hist_maxinterval(h, 1.0 - self.graph.significance*2) if h is not False else (mean, mean)
            if sg > self.graph.significance:
                self.ndcond += [cond]
                if self.status == 'Direct':
                    self.status = 'Indirect'
                #if not silent:
                print("H2: Confirming [" + self.x1 + "," + self.x2 + "," + self.x3 + "] " + 'Non-direct')


class CausalModel():
    def __init__(self):
        self.count = 0
        self.data = []
        self.header = None
        self.form = None
        self.significance = 0.01
        self.tolerance = 0.0
        self.citest = 'gsam'
        self.highorder = False
        self.nodes = []
        self.links = []
        self.hlinks = []
        self.reserved_attrs = []
        self.version = -1
        self.type = 'CausalModel'

    def reset_data(self):
        self.count = 0
        self.data = []
        self.header = None
        self.form = None

    def handle_data(self, vecdict):
        if self.header is None:
            self.header = [key for key in vecdict if key not in self.reserved_attrs]
            self.create_session()
            self.version += 1
        vec = [vecdict[key] for key in self.header]
        self.data += [vec]
        self.count += 1
        # at regular intervals, call update_model
        if self.count % 200 == 150:
            self.update_model()
        return None

    def handle_batch_data(self, dt):
        if len(dt)>0:
            if self.header is None:
                self.header = [key for key in dt[0] if key not in self.reserved_attrs]
                self.create_session()
                self.version += 1
            newdata = [[dt[i][key] for key in self.header] for i in range(len(dt))]
            self.data += newdata
            self.count += len(newdata)
            self.update_model()
        return None

    def model_type(self):
        return self.type

    def model_version(self):
        return self.version

    def extract_model(self):
        return { 'type': self.type,
                 'version': self.version,
                 'nodes': self.nodes,
                 'links': self.links,
                 'hlinks': self.hlinks }

    def features_changed(self):
        return False

    def default_params(self):
        return { 'significance': 0.01,
                 'tolerance': 0.0,
                 'citest': 'gsam',
                 'highorder': False,
                 'index_attribute': False,
                 'time_attribute': False,
                 'entity_attribute': False}

    def set_params(self, dic):
        if 'significance' in dic:
            self.significance = dic['significance']
        if 'tolerance' in dic:
            self.tolerance = dic['tolerance']
        if 'citest' in dic:
            self.citest = dic['citest']
        if 'highorder' in dic:
            self.highorder = dic['highorder']
        if 'index_attribute' in dic:
            self.reserved_attrs.append(dic['index_attribute'])
        if 'time_attribute' in dic:
            self.reserved_attrs.append(dic['time_attribute'])
        if 'entity_attribute' in dic:
            self.reserved_attrs.append(dic['entity_attribute'])

    #-------------------------------------------

    def create_session(self):
        self.links = []
        self.nodes = []
        self.hlinks = []
        for name in self.header:
            self.nodes.append(Node(self, name))
        for pair in subsets(self.nodes,2):
            self.links.append(Edge(self, pair[0].name, pair[1].name))
        if self.highorder:
            for tr in subsets(self.nodes, 3):
                self.hlinks.append(self, HyperEdge(tr[0].name, tr[1].name, tr[2].name))

    def update_model(self):
        self.form = make_format(self.data, ['cont' for i in range(len(self.header))])
        for n in self.nodes:
            n.update_prob()
        for e in self.links:
            e.reset()
            e.update_indirect()
        for e in self.links:
            if e.status == 'Indirect':
                e.status = 'Direct'
                e.dir = e.indir
        for i in range(1,4):
            for e in self.links:
                e.update_direct(i)
        for e in self.links:
            if not e.status == e.oldstatus:
                e.changed = True
                #if not silent:
                print("3: Making [" + e.x1 + "," + e.x2 + "] " + e.status)
        self.convergers = []
        for n in self.nodes:
            self.update_causal_1(n.name)
        self.update_causal_2()
        for n in self.nodes:
            self.update_causal_strengths(n.name)
        if self.highorder:
            for e in self.hlinks:
                e.reset()
                e.update_indirect()
            for e in self.hlinks:
                if e.status == 'Indirect':
                    e.status = 'Direct'
                    e.dir = e.indir
            for lev in range(1,3):
                for e in self.hlinks:
                    e.update_direct(lev)
            for e in self.hlinks:
                if not e.status == e.oldstatus:
                    e.changed = True
                    #if not silent:
                    print("H3: Making [" + e.x1 + "," + e.x2 + "," + e.x3 + "] " + e.status)
        self.version += 1

    def update_causal_1(self, name):
        # hitta alla länkar vars båda ändar sitter ihop med name
        # om någon av dem är Indirect men saknar name i ndcond -> båda pilar in
        ns = subsets(find_linked_nodes(self.links, name), 2)
        for e in self.links:
            if [e.x1, e.x2] in ns and e.status in ['Indirect','Unconfirmed'] and not deepmem(name, e.ndcond):
                sg, mean, h = signif(self.data,
                                     list(map(lambda n:find_node_index(self.nodes, n), [e.x1, e.x2])),
                                     list(map(lambda n:find_node_index(self.nodes, n), [name] + (e.ndcond[0] if not e.ndcond == [] else []))),
                                     self.citest, self.form, self.tolerance)
                if sg < self.significance:
                    ok = True
                    for ele in self.convergers:
                        if ((ele[0] == e.x1 or ele[0] == e.x2) and
                            (name == ele[1] or name == ele[2])):
                            ele[3] = False
                            ok = False
                    self.convergers.append([name, e.x1, e.x2, ok])

    def update_causal_2(self):
        for ele in self.convergers:
            (name, x1, x2, ok) = tuple(ele)
            if ok:
                print("4: Converging to " + name + ": " + x1 + " and " + x2)
                ee = find_link(self.links, name, x1)
                ee.status = 'Causal'
                ee.forward = name == ee.x2
                ee = find_link(self.links, name, x2)
                ee.status = 'Causal'
                ee.forward = name == ee.x2
                #if not silent:
            else:
                print("4: Ambiguous to " + name + ": " + x1 + " and " + x2)

    def update_causal_strengths(self, name):
        # Uppdatera direkt styrka
        # Hitta länkar (och deras noder) som pekar mot name, betinga på dem
        cond = []
        lks = []
        nidx = find_node_index(self.nodes, name)
        for e in self.links:
            if e.status == 'Causal':
                if e.x1 == name and not e.forward:
                    cond += [find_node_index(self.nodes, e.x2)]
                    lks += [e]
                elif e.x2 == name and e.forward:
                    cond += [find_node_index(self.nodes, e.x1)]
                    lks += [e]
        for e in lks:
            midx = find_node_index(self.nodes, e.x1 if e.forward else e.x2)
            sg, mean, h = signif(self.data,
                                 [midx, nidx],
                                 [i for i in cond if not i==midx],
                                 self.citest, self.form, self.tolerance)
            mn, mx = hist_maxinterval(h, 1.0 - self.significance*2) if h is not False else (mean, mean)
            e.dir = (mean, mn, mx, sg) # safesqrt(var),
            e.hist = h


class PlainAttributesModel():
    def __init__(self):
        self.header = None
        self.nodes = []
        self.reserved_attrs = []
        self.version = -1
        self.type = 'PlainAttributesModel'

    def reset_data(self):
        self.header = None

    def handle_data(self, vecdict):
        if self.header is None:
            self.header = [key for key in vecdict if key not in self.reserved_attrs]
            self.create_nodes()
            self.version += 1
        return None

    def handle_batch_data(self, dt):
        if len(dt)>0:
            if self.header is None:
                self.header = [key for key in dt[0] if key not in self.reserved_attrs]
                self.create_nodes()
                self.version += 1
        return None

    def model_type(self):
        return self.type

    def model_version(self):
        return self.version

    def extract_model(self):
        return { 'type': self.type,
                 'version': self.version,
                 'nodes': self.nodes,
                 'links': [],
                 'hlinks': [] }

    def features_changed(self):
        return False

    def default_params(self):
        return { 'index_attribute': False,
                 'time_attribute': False,
                 'entity_attribute': False}

    def set_params(self, dic):
        if 'index_attribute' in dic:
            self.reserved_attrs.append(dic['index_attribute'])
        if 'time_attribute' in dic:
            self.reserved_attrs.append(dic['time_attribute'])
        if 'entity_attribute' in dic:
            self.reserved_attrs.append(dic['entity_attribute'])

    #-------------------------------------------

    def create_nodes(self):
        self.nodes = []
        for name in self.header:
            self.nodes.append(Node(self, name))
