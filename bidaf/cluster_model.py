from math import * 
import pandas as pd

import matplotlib.pyplot as plt

from Samecolor import *
from Color import *
from hgm import *
from featurename import *

class ClusterModel():
    def __init__(self):
        self.count = 0
        self.data = []
        self.timeent = []
        self.header = None
        self.gmm = False
        self.prior = False
        self.lasty = [0]
        self.perm = {0:0}
        self.changed = False
        self.version = -1
        self.type = 'ClusterModel'
        self.featurename1 = 'ClusterIndex'
        self.featurename2 = 'ClusterAnomaly'
        self.index_attribute = False
        self.time_attribute = False
        self.entity_attribute = False
        self.maxsplit = 1

    def dictaddindex(self, vec, dic):
        res = {}
        if self.time_attribute:
            res[self.time_attribute] = vec[self.time_attribute]
        else:
            res[self.index_attribute] = vec[self.index_attribute]
        if self.entity_attribute:
            res[self.entity_attribute] = vec[self.entity_attribute]
        res.update(dic)
        return res

    def indexvalues(self, vec):
        return tuple([vec[a] for a in self.indexattrs()])

    def indexattrs(self):
        return ([self.index_attribute if not self.time_attribute else self.time_attribute] +
                ([self.entity_attribute] if self.entity_attribute else []))

    def reset_data(self):
        self.count = 0
        self.data = []
        self.timeent = []
        self.header = None

    def handle_data(self, vecdict):
        if self.header is None:
            self.header = [key for key in vecdict if key not in self.indexattrs()]
            form = { col : ContForm(0,0) for col in self.header}
            self.gmm = MixtureModel(GaussianModel(self.header, form), 1)
            self.prior = MixturePrior(self.gmm, 1.0, form, pd.DataFrame(columns=self.header))
            self.gmm.estimate_init(self.prior)
        vec = pd.DataFrame([vecdict]).iloc[0]
        self.data.append(vec)
        self.timeent.append(self.indexvalues(vecdict))
        self.gmm.estimate_incr(vec, 1.0)
        self.count += 1
        # at regular intervals, call update_model
        if self.count % 200 == 50:
            self.maxsplit = 1
            self.update_model()
        return self.dictaddindex(vecdict,
                                 {self.featurename1: self.perm[self.maxindex(vec)],
                                  self.featurename2: self.anomaly(vec)})

    def handle_batch_data(self, dt):
        if len(dt) > 0:
            newdata = pd.DataFrame(dt)
            self.data += [newdata.iloc[i] for i in range(len(newdata))]
            self.timeent += [self.indexvalues(newdata.iloc[i]) for i in range(len(newdata))]
            self.count += len(newdata)
            if self.header is None:
                self.header = [key for key in dt[0] if key not in self.indexattrs()]
                form = { col : ContForm(0,0) for col in self.header}
                self.gmm = MixtureModel(GaussianModel(self.header, form), 1)
                self.prior = MixturePrior(self.gmm, 1.0, form, pd.DataFrame(self.data, columns=self.header))
                self.gmm.estimate_init(self.prior)
            self.maxsplit = 6
            self.update_model()
            return [self.dictaddindex(newdata.iloc[i],
                                      {self.featurename1: self.perm[self.maxindex(newdata.iloc[i])],
                                       self.featurename2: self.anomaly(newdata.iloc[i])}) for i in range(len(newdata))]
        else:
            return None

    def model_type(self):
        return self.type

    def model_version(self):
        return self.version

    def extract_model(self):
        if self.gmm is not False:
            mod = [{ 'prob':p, 'mean':m.mean, 'var':m.var} for m,p in zip(self.gmm.models,self.gmm.probs)]
        else:
            mod = []
        return { 'type': self.type,
                 'version': self.version,
                 'mixture': mod }

    def features_changed(self):
        return self.changed

    def maxindex(self, sample):
        if len(self.gmm.models) == 1:
            return 0
        else:
            v = [m.logprobability(sample) + log(p) for m,p in zip(self.gmm.models,self.gmm.probs)]
            return v.index(max(v))

    def anomaly(self, sample):
        #if self.gmm.count < 2*len(sample):
        if self.version < 0: # initialized
            return 0.0
        else:
            ano = self.gmm.anomaly(sample)
            return min(1.0, max(0.0, (ano - 14.0)/36.0))

    def update_features(self):
        self.changed = False
        y = [self.maxindex(v) for v in self.data]
        z = [self.anomaly(v) for v in self.data]
        ia = self.indexattrs()
        return [self.dictaddindex(dict(zip(ia, te)),
                                  {self.featurename1: self.perm[yy],
                                   self.featurename2: zz}) for (te,yy,zz) in zip(self.timeent, y, z)] 
        
    def default_params(self):
        return { 'index_attribute': False,
                 'time_attribute': False,
                 'entity_attribute': False}

    def set_params(self, dic):
        #if 'num_components' in dic:
        #    self.numcomp = dic['num_components']
        if 'time_attribute' in dic:
            self.time_attribute = dic['time_attribute']
        if 'index_attribute' in dic:
            self.index_attribute = dic['index_attribute']
        if 'entity_attribute' in dic:
            self.entity_attribute = dic['entity_attribute']

    #----------------------------

    def update_model(self):
        def smaxind(ss, i):
            v = [ss[j][i] for j in range(len(ss))]
            return v.index(max(v))
        # check if split, if so run em
        df = pd.DataFrame(self.data, columns= self.header)
        if not self.gmm.initialized:
            ss = self.gmm.init_ss(df, False)
            mod.initialized = True
        else:
            ss = self.gmm.get_ss(df, False)
        # ok rebuild the prior
        self.prior = MixturePrior(self.gmm, 1.0, self.gmm.form, df)
        #print("Old: ", self.gmm.models[0].mean, self.gmm.models[0].var)
        ss = self.gmm.estimate_ss(df, False, self.prior, ss, 20, 0.001)
        #print("New: ", self.gmm.models[0].mean, self.gmm.models[0].var)
        loop = 0
        while loop < self.maxsplit:
            sigs = []
            if self.time_attribute is not False:
                scales = [feature_scale(n) for n in self.header]
                maxtimescale = max(scales) if scales else 15.0
                tmmd = 4*maxtimescale
                print("Gamma timescale:", tmmd)
            else:
                tmmd = nan
            for i in range(len(ss)):
                sigs.append(gammatest_ts(self.gmm.models[i], ss[i], df, self.timeent, 4000, tmmd))
            print("Gamma sigs: ", sigs)
            mx = min(sigs)
            if mx < 0.001:
                ss = self.gmm.split(sigs.index(mx), ss)
                ss = self.gmm.estimate_ss(df, False, self.prior, ss, 100, 0.00001)
                y = [smaxind(ss,i) for i in range(len(self.data))]
                self.perm = samecolor(self.lasty, y, self.perm)
                self.lasty = y
                loop += 1
            else:
                loop = self.maxsplit
        self.version += 1
        self.changed = True
        

def gammatest_ts(comp, weights, data, timeent, num, tsmindist):
    # set up vector for quicker search
    # (select first sample)
    # a large number of times
    #   select random sample
    #   calculate normalized distance from last sample
    # sort distances
    # express expected gamma parameters
    # find the distance at which deviation from expected distr is largest
    # compute its significane level
    data = data[comp.feet]
    dim = len(comp.feet)
    acc = np.cumsum(weights)
    dists = [False]*num
    ps = [False]*num
    oldind = selectind(acc)
    ind = oldind
    wsum = 0
    anum = 0
    if acc[-1] < 10.0 or len(data) < 2*sqrt(num) or (not isnan(tsmindist) and (timeent[-1][0] - timeent[0][0]) < tsmindist):
        return 1.0
    for i in range(num):
        while ind == oldind:
            ind = selectind(acc)
        if isnan(tsmindist) or (len(timeent[ind]) >= 2 and timeent[ind][1] != timeent[oldind][1]) or abs(timeent[ind][0] - timeent[oldind][0]) >= tsmindist:
            dist = sqvec(np.matmul(comp.ilower, np.subtract(data.iloc[ind], data.iloc[oldind])))
            w = weights[oldind] # Because weights[ind] is already accounted for in selection
            dists[i] = (dist, w)
            wsum += w
            anum += 1
        else:
            dists[i] = (inf, 0.0)
        oldind = ind
    dists.sort(key = lambda pr:pr[0])
    mnval = 0
    mnind = 0
    w = 0
    for i in range(anum):
        (d, w0) = dists[i]
        w += w0
        p = gammainc(dim/2, d/4)
        ps[i] = p
        # print((wsum, w, p))
        val = log(wsum*p/w)*w + log(wsum*(1-p)/(wsum-w))*(wsum-w) if p > 0 and p < 1 and w > 0 and w < wsum else 0
        dists[i] = (d, w) 
        if val<mnval and p<0.95 and p>0.05 and p*wsum > w:
            mnval = val
            mnind = i
    (d, w) = dists[mnind]
    p = ps[mnind] #gammainc(dim/2, d/4)
    if mnval == 0:
        sig = 1.0
    elif w > wsum*p:
        n = ceil(w)
        nn = ceil(wsum)
        sig = binom.cdf((nn-n), nn, (1-p))
    else:
        n = floor(w)
        nn = ceil(wsum)
        sig = binom.cdf(n, nn, p)

    #plt.figure(2)
    #plt.axes().clear()
    #plt.plot([dists[i][0] for i in range(anum)],[dists[i][1] for i in range(anum)])
    #plt.plot([dists[i][0] for i in range(anum)],[ps[i]*wsum for i in range(anum)])
    
    return sig


