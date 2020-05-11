from math import * 
from random import *
import pandas as pd
import numpy as np
from scipy.special import gammainc,gammaincc
from scipy.stats import binom

# Data object: pd.dataframe
# Format object: dictionary { name : Formspec }


#----------------------------------------------------------
# Misc functions
#----------------------------------------------------------

def incrlistpart(lst, args, incr):
    for ind in args[:-1]:
        lst = lst[ind]
    lst[args[-1]] += incr

def getlistpart(lst, args):
    for ind in args[:-1]:
        lst = lst[ind]
    return lst[args[-1]]

def depth(a):
    return 1 + depth(a[0]) if type(a) is list else 0

def deepcopy(lst):
    return list(map(deepcopy, lst)) if type(lst)==list else lst

def deepsum(cube):
    if type(cube[0]) == list:
        return sum([deepsum(lst) for lst in cube])
    else:
        return sum(cube)

def deepscale(cube, scale):
    if type(cube[0]) == list:
        return [deepscale(lst, scale) for lst in cube]
    else:
        return list(map(lambda v: v*scale, cube))

def deeptimespower(cubelst, powlst):
    if type(cubelst[0]) == list:
        return [deeptimespower([c[i] for c in cubelst], powlst) for i in range(len(cubelst[0]))]
    else:
        return times(map(pow, cubelst, powlst))

def deeprandompick(cube, thres):
    if type(cube[0]) == list:
        for i in range(len(cube)):
            s = deepsum(cube[i])
            if s>thres:
                return [i] + deeprandompick(cube[i], thres)
            else:
                thres -= s
    else:
        for i in range(len(cube)):
            if cube[i]>thres:
                return [i]
            else:
                thres -= cube[i]

def deepselect(cube, sel):
    if type(cube) == list:
        return [deepselect(cube[i], sel) for i in sel]
    else:
        return cube

def makecube(dims, val):
    if dims != []:
        sub = makecube(dims[1:], val)
        lst = list(range(dims[0]))
        lst[0] = sub
        for i in range(1, dims[0]):
            lst[i] = deepcopy(sub)
        return lst
    else:
        return val

def cubelevelsum(cube):
    if type(cube[0]) == list:
        return [cubelevelsum(list(map(lambda lst:lst[i], cube))) for i in range(len(cube[0]))]
    else:
        return sum(cube)

def cubemargin(cube, marg):
    if type(cube)==list:
        if depth(cube)-1 in marg:
            return list(map(lambda lst: cubemargin(lst, marg), cube))
        else:
            return cubemargin(cubelevelsum(cube), marg)
    else:
        return cube

def cubeweightedsum(cubelst, wlst):
    if type(cubelst[0]) == list:
        return [cubeweightedsum([c[j] for c in cubelst], wlst) for j in range(len(cubelst[0]))]
    else:
        return sum(map(lambda c,w: c*w, cubelst, wlst))

def subcube(cube, vec):
    if len(vec)==0:
        return cube
    elif np.isnan(vec[0]):
        return [subcube(c, vec[1:]) for c in cube]
    else:
        return subcube(cube[vec[0]], vec[1:])

def accmatrix(matrix, vec, weight):
    rng = range(len(vec))
    for i in rng:
        for j in rng:
            matrix[i][j] += vec[i]*vec[j]*weight

def flatten(lst):
    return list(np.array(lst).flat)

def flattenuniq(lst):
    res = []
    for e1 in lst:
        for e2 in e1:
            if not e2 in res:
                res.append(e2)
    return res

def times(lst):
    prod = 1.0
    for x in lst:
        prod *= x
    return prod

def sqvec(vec):
    return np.dot(vec,vec)

def normalizelogs(lst, w):
    if w==0.0:
        return [0.0 for x in lst]
    else:
        mx = max(lst)
        tmp = [exp(x - mx) for x in lst]
        sm = sum(tmp)/w
        return [x/sm for x in tmp]

def normalizelogsprior(lst, w, prw):
    if w==0.0:
        return [0.0 for x in lst]
    else:
        mx = max(lst)
        prwn = prw/len(lst)
        tmp = [exp(x - mx) for x in lst]
        sm = sum(tmp)/w*(1.0+prw)
        return [x/sm + prwn for x in tmp]

def normalize(lst):
    sm = sum(lst)
    return [e/sm for e in lst]

def issubset(s1, s2):
    if len(s1) >= len(s2):
        return False
    for e in s1:
        if not e in s2:
            return False
    return True

def hascommon(s1, s2):
    for e in s1:
        if e in s2:
            return True
    return False

#----------------------------------------------------------
# Base classes
#----------------------------------------------------------

class Model:
    def __init__(self, feet, form):
        self.feet = feet
        self.form = form
        self.count = 0.0

class Prior:
    def __init__(self, alpha):
        self.alpha = alpha

class Formspec:
    def __init__(self):
        pass

class ContForm(Formspec):
    def __init__(self, mn, mx):
        self.min = mn
        self.max = mx

    def interpret(self, val):
        return val

    def represent(self, ival):
        return ival

class DiscrForm(Formspec):
    def __init__(self, num, offset = 0):
        self.num = num
        self.offset = offset

    def interpret(self, val):
        return val - self.offset if not np.isnan(val) else np.nan

    def represent(self, ival):
        return ival + self.offset if not np.isnan(ival) else np.nan

class SymbolForm(Formspec):
    def __init__(self, vals):
        self.values = vals
        self.num = len(vals)

    def interpret(self, val):
        if val in self.values:
            return self.values.index(val)
        else:
            return np.nan

    def represent(self, ival):
        if ival >= 0 and ival < len(self.values):
            return self.values[ival]
        else:
            return np.nan

#----------------------------------------------------------
# BernoulliModel
#----------------------------------------------------------

class BernoulliModel(Model):
    def __init__(self, feet, form, prob = False, count = False):
        Model.__init__(self, feet, form)
        if prob is not False:
            self.prob = deepcopy(prob)
            self.count = count
        else:
            self.prob = makecube([self.form[f].num for f in self.feet], 0.0)
            self.count = 0.0

    def copy_struct(self):
        return BernoulliModel(self.feet, self.form)

    def estimate(self, data, weights, prior):
        counts = deepscale(prior.cube, prior.alpha)
        if weights is not False:
            for (d,w) in zip(data[self.feet].values, weights):
                incrlistpart(counts, d, w)
        else:
            for d in data[self.feet].values:
                incrlistpart(counts, d, 1.0)
        self.count = deepsum(counts)
        self.prob = deepscale(counts, 1.0/self.count if self.count>0 else 0.0)

    def estimate_init(self, prior):
        counts = deepscale(prior.cube, prior.alpha)
        self.count = deepsum(counts)
        self.prob = deepscale(counts, 1.0/self.count if self.count>0 else 0.0)

    def estimate_incr(self, sample, weight):
        if weight != 0.0:
            self.prob = deepscale(self.prob, self.count/(self.count+weight))
            incrlistpart(self.prob, sample[self.feet], weight/(self.count+weight))
            self.count += weight
        
    def generate(self, sample):
        lst = deeprandompick(self.prob, random())
        for i in range(len(self.feet)):
            sample[self.feet[i]] = lst[i]
        return sample

    def probability(self, sample):
        return getlistpart(self.prob, [sample[f] for f in self.feet])

    def logprobability(self, sample):
        return log(self.probability(sample))

    def logmodelprob(self, prior):
        return sum([(a*prior.alpha)*log(p) if p>0 else 0 for (p,a) in zip(flatten(self.prob), flatten(prior.cube))])

    def margin(self, mlst):
        l = len(self.feet)
        flst = [f for f in self.feet if f in mlst]
        clst = [l-i-1 for i in range(l) if self.feet[i] in mlst]
        return BernoulliModel(flst, self.form, cubemargin(self.prob, clst), self.count)

    def predict(self, sample):
        vec = [sample[f] for f in self.feet]
        flst = [f for f in self.feet if np.isnan(sample[f])]
        cube = subcube(self.prob, vec)
        s = sum(flatten(cube))
        return BernoulliModel(flst, self.form, deepscale(cube, 1.0/s if s>0 else 0.0), self.count/s)

    def anomaly(self, sample):
        p = getlistpart(self.prob, [sample[f] for f in self.feet])
        cump = sum([pp for pp in flatten(cube) if pp <= p])
        return -log(cump)

    def multiply(self, modlst, cntlst):
        cube = deeptimespower([m.prob for m in modlst], cntlst)
        s = sum(flatten(cube))
        return BernoulliModel(self.feet, self.form, deepscale(cube, 1.0/s if s>0 else 0.0), min([m.count for m in modlst]))

    def average(self, modlst):
        wsum = sum([m.count for m in modlst])
        cube = cubeweightedsum([m.prob for m in modlst],[m.count/wsum for m in modlst])
        return BernoulliModel(self.feet, self.form, cube, wsum)

    def prior(self, alpha, data):
        return BernoulliPrior(self, alpha, self.form, data)

class BernoulliPrior(Prior):
    def __init__(self, model, alpha, form, data):
        Prior.__init__(self, alpha)
        n = 1
        for f in model.feet:
            n *= form[f].num
        self.cube = makecube([form[f].num for f in model.feet], 1.0/n)


#----------------------------------------------------------
# GaussianModel
#----------------------------------------------------------

class GaussianModel(Model):
    def __init__(self, feet, form, mean = False, var = False, count = False):
        Model.__init__(self, feet, form)
        self.mean = mean
        self.var = var
        self.count = count
        self.upper = False
        self.ilower = False
        self.logidet = False

    def copy_struct(self):
        return GaussianModel(self.feet, self.form)

    def estimate(self, data, weights, prior):
        num = prior.alpha
        sum = [prior.mean[i] * prior.alpha for i in range(len(self.feet))]
        if weights is not False:
            for (d,w) in zip(data[self.feet].values, weights):
                num += w
                for i in range(len(self.feet)):
                    sum[i] += d[i] * w
        else:
            for d in data[self.feet].values:
                num += 1.0
                for i in range(len(self.feet)):
                    sum[i] += d[i]
        self.count = num
        self.mean = [sum[i]/num if num>0.0 else 0.0 for i in range(len(self.feet))]
        var = makecube([len(self.feet),len(self.feet)], 0.0)
        for i in range(len(self.feet)):
            var[i][i] = prior.var[i][i] * prior.alpha
        if weights is not False:
            for (d,w) in zip(data[self.feet].values, weights):
                accmatrix(var, [d[i]-self.mean[i] for i in range(len(self.feet))], w)
        else:
            for d in data[self.feet].values:
                accmatrix(var, [d[i]-self.mean[i] for i in range(len(self.feet))], 1.0)
        accmatrix(var, [prior.mean[i]-self.mean[i] for i in range(len(self.feet))], prior.alpha)
        var = deepscale(var, 1.0/num if num>0.0 else 0.0)
        self.var = var
        self.upper = False
        self.ilower = False
        self.logidet = False

    def estimate_init(self, prior):
        self.count = prior.alpha
        self.mean = [prior.mean[i] for i in range(len(self.feet))]
        self.var = makecube([len(self.feet),len(self.feet)], 0.0)
        for i in range(len(self.feet)):
            self.var[i][i] = prior.var[i][i]
        self.upper = False
        self.ilower = False
        self.logidet = False

    def estimate_incr(self, sample, weight):
        #for i in range(len(self.feet)):
        #    self.mean[i] += (sample[self.feet[i]] - self.mean[i]) * weight/(self.count + weight)
        #self.var = deepscale(self.var, self.count/(self.count + weight))
        #accmatrix(self.var, [sample[self.feet[i]] - self.mean[i] for i in range(len(self.feet))], weight/(self.count + weight))
        if weight > 0.0:
            deltamean = [sample[self.feet[i]] - self.mean[i] for i in range(len(self.feet))]
            for i in range(len(self.feet)):
                self.mean[i] += deltamean[i] * weight/(self.count + weight)
            self.var = deepscale(self.var, self.count/(self.count + weight))
            accmatrix(self.var, deltamean, self.count*weight/(self.count + weight)/(self.count + weight))
            self.count += weight
        self.upper = False
        self.ilower = False
        self.logidet = False

    def prepare(self):
        tmp = np.linalg.cholesky(self.var)
        self.upper = np.transpose(tmp)
        self.ilower = np.linalg.inv(tmp)
        self.logidet = log(times(np.diagonal(self.ilower)))

    def generate(self, sample):
        if self.upper is False:
            self.prepare()
        vec = np.add(self.mean, np.matmul([gauss(0,1) for i in range(len(self.feet))], self.upper))
        for i in range(len(self.feet)):
            sample[self.feet[i]] = vec[i]
        return sample

    def probability(self, sample):
        return exp(self.logprobability(sample))

    def logprobability(self, sample):
        if self.upper is False:
            self.prepare()
        v = [sample[self.feet[i]] - self.mean[i] for i in range(len(self.feet))]
        return self.logidet - len(self.feet)*log(2*pi)/2 - sqvec(np.matmul(self.ilower, v))/2

    def logmodelprob(self, prior):
        if self.upper is False:
            self.prepare()
        pass

    def margin(self, mlst):
        flst = [f for f in self.feet if f in mlst]
        clst = [i for i in range(len(self.feet)) if self.feet[i] in mlst]
        return GaussianModel(flst, self.form, deepselect(self.mean, clst), deepselect(self.var, clst), self.count)

    def predict(self, sample):
        flst = [f for f in self.feet if np.isnan(sample[f])]
        ulst = [i for i in range(len(self.feet)) if np.isnan(sample[self.feet[i]])]
        klst = [i for i in range(len(self.feet)) if not np.isnan(sample[self.feet[i]])]
        if len(ulst)==0:
            return GaussianModel([], self.form, [], [[]], 0.0)
        elif len(klst)==0:
            return self
        else:
            vec = [sample[f] for f in self.feet if not np.isnan(sample[f])]
            nmeanu = [self.mean[i] for i in ulst]
            nmeank = [self.mean[i] for i in klst]
            nvar = [[self.var[i][j] for j in klst+ulst] for i in klst+ulst]
            tmp = np.linalg.cholesky(nvar)
            nupper = np.transpose(tmp)
            nilower = np.linalg.inv(tmp)
            b = [[nilower[i][j] for j in klst] for i in ulst]
            c = [[nupper[i][j] for j in ulst] for i in ulst]
            nmean = nmeanu - np.matmul(np.transpose(c), np.matmul(b, [vec[i] - nmeank[i] for i in range(len(klst))]))
            nvar = np.matmul(np.transpose(c), c)
            return GaussianModel(flst, self.form, nmean, nvar, self.count)

    def anomaly(self, sample):
        if self.upper is False:
            self.prepare()
        v = [sample[self.feet[i]] - self.mean[i] for i in range(len(self.feet))]
        md = sqvec(np.matmul(self.ilower, v))/2
        gi = gammaincc(len(self.feet)/2, md)
        return -np.log(gi) if gi > 0.0 else 1000.0
        
    def multiply(self, modlst, cntlst):
        invs = [np.linalg.inv(m.var) for m in modlst]
        sinv = sum([invs[i]*cntlst[i] for i in range(len(modlst))])
        sres = np.linalg.inv(sinv)
        means = [m.mean for m in modlst]
        smean = sum([np.matmul(invs[i],means[i])*cntlst[i] for i in range(len(modlst))])
        mres = np.matmul(sres, smean)
        return GaussianModel(self.feet, self.form, mres, sres, min([m.count for m in modlst]))

    def average(self, modlst):
        counts = [m.count for m in modlst]
        count = sum(counts)
        means = [m.mean for m in modlst]
        mres = deepscale(cubeweightedsum(means, counts), 1.0/count if count>0.0 else 0.0)
        vars = [m.var for m in modlst]
        vres = cubeweightedsum(vars, counts)
        if count > 0.0:
            for i in range(len(means)):
                accmatrix(vres, [means[i][j] - mres[j] for j in range(len(means[0]))], counts[i])
            vres = deepscale(vres, 1.0/count)
        return GaussianModel(self.feet, self.form, mres, vres, count)


    def prior(self, alpha, data):
        return GaussianPrior(self, alpha, self.form, data)

    
class GaussianPrior(Prior):
    def __init__(self, model, alpha, form, data):
        Prior.__init__(self, alpha)
        self.feet = model.feet
        self.estimate(data)

    def estimate(self, data):
        num = 0.0
        sum = [0.0 for i in range(len(self.feet))]
        qsum = [0.0 for i in range(len(self.feet))]
        for d in data[self.feet].values:
            num += 1.0
            for i in range(len(self.feet)):
                sum[i] += d[i]
        self.mean = [sum[i]/num if num>0.0 else 0.0 for i in range(len(self.feet))]
        self.var = makecube([len(self.feet),len(self.feet)], 0.0)
        for d in data[self.feet].values:
            for i in range(len(self.feet)):
                qsum[i] += (d[i]-self.mean[i])*(d[i]-self.mean[i])
        if num > 0.0:
            for i in range(len(self.feet)):
                tmp = qsum[i]/num
                self.var[i][i] = tmp if tmp > 0.0 else 1.0
        else:
            for i in range(len(self.feet)):
                self.var[i][i] = 1.0
                

#----------------------------------------------------------
# ProductModel
#----------------------------------------------------------

class ProductModel(Model):
    def __init__(self, models):
        Model.__init__(self, flattenuniq([m.feet for m in models]), models[0].form if models is not [] else {})
        self.models = models
        self.count = max([m.count for m in models])

    def estimate(self, data, weights, prior):
        for i in range(len(self.models)):
            self.models[i].estimate(data, weights, prior.priors[i])
        self.count = max([m.count for m in self.models])

    def estimate_init(self, prior):
        for i in range(len(self.models)):
            self.models[i].estimate_init(prior.priors[i])
        self.count = prior.alpha

    def estimate_incr(self, sample, weight):
        for i in range(len(self.models)):
            self.models[i].estimate_incr(sample, weight)
        self.count += weight

    def generate(self, sample):
        for i in range(len(self.models)):
            self.models[i].generate(sample)
        return sample

    def probability(self, sample):
        return times(map(lambda m: m.probability(sample), self.models))

    def logprobability(self, sample):
        return sum(map(lambda m: m.logprobability(sample), self.models))

    def logmodelprob(self, prior):
        sum = 0.0
        for i in range(len(self.models)):
            sum += self.models[i].logmodelprob(prior.priors[i])
        return sum

    def margin(self, mlst):
        modlst = [m.margin(mlst) for m in self.models if hascommon(m.feet,mlst)]
        return ProductModel(modlst)

    def predict(self, sample):
        flst = [f for f in self.feet if np.isnan(sample[f])]
        modlst = [m.predict(sample) for m in self.models if hascommon(m.feet,flst)]
        return ProductModel(modlst)

    def anomaly(self, sample):
        # Approximation: just sum up anomalies of factors
        return sum(map(lambda m: m.anomaly(sample), self.models))

    def multiply(self, modlst, cntlst):
        submodels = [m.models for m in modlst]
        nmodlst = [submodels[0][i].multiply([submodels[j][i] for j in range(len(modlst))], cntlst) for i in range(len(submodels[0]))]
        return ProductModel(nmodlst)

    def average(self, modlst):
        submodels = [m.models for m in modlst]
        nmodlst = [submodels[0][i].average([submodels[j][i] for j in range(len(modlst))]) for i in range(len(submodels[0]))]
        return ProductModel(nmodlst)
        
    def prior(self, alpha, data):
        return ProductPrior(self, alpha, self.form, data)

    def copy_struct(self):
        return ProductModel([m.copy_struct() for m in self.models])


class ProductPrior(Prior):
    def __init__(self, model, alpha, form, data):
        Prior.__init__(self, alpha)
        self.priors = [m.prior(alpha, data) for m in model.models]


#----------------------------------------------------------
# MixtureModel
#----------------------------------------------------------

em_min_diff = 0.000001
em_max_iter = 1000
em_init_size = 128

global_ss = False

def set_min_diff(val):
    global em_min_diff
    em_min_diff = val

def set_max_iter(val):
    global em_max_iter
    em_max_iter = val

def set_init_size(val):
    global em_init_size
    em_init_size = val

def get_ss():
    return global_ss

class MixtureModel(Model):
    def __init__(self, mod, num_or_probs):
        if type(mod)==list and type(num_or_probs)==list:
            Model.__init__(self, mod[0].feet, mod[0].form)
            self.models = mod
            self.probs = num_or_probs
            self.count = sum([m.count for m in mod])
            self.initialized = True
        else:
            Model.__init__(self, mod.feet, mod.form)
            self.models = [mod.copy_struct() for i in range(num_or_probs)]
            self.probs = [1.0/num_or_probs for i in range(num_or_probs)]
            self.initialized = False

    def estimate(self, data, weights, prior, extss = False):
        global global_ss
        # initiera komponenttillhörighet
        # E: gör estimate på varje komponent
        # M: beräkna nya tillhörigheter
        # kolla förändringen (och likelihood och antal iter)
        dnum = len(data)
        knum = len(self.models)
        ss = [[0.0 for i in range(dnum)] for j in range(knum)]
        if not self.initialized:
            if extss is not False:
                for j in range(knum):
                    for i in range(dnum):
                        ss[j][i] = extss[j][i] if weights is False else extss[j][i]*weights[i]
            elif dnum < knum*em_init_size:
                for i in range(dnum):
                    ss[randint(0,knum-1)][i] = 1.0 if weights is False else weights[i]
                global_ss = pd.DataFrame(np.transpose(ss))
            elif weights is not False:
                rng = list(range(dnum))
                for j in range(knum*em_init_size):
                    i = choices(rng, weights)
                    k = randint(0,knum-1)
                    if k >= knum or i >= dnum:
                        print("k=%d, i=%d" % (k,i))
                    else:
                        ss[k][i] = weights[i]
            else:
                for j in range(knum*64):
                    i = randint(0,dnum-1)
                    k = randint(0,knum-1)
                    if k >= knum or i >= dnum:
                        print("k=%d, i=%d" % (k,i))
                    else:
                        ss[k][i] = 1.0
            self.initialized = True
        else:
            for i in range(dnum):
                v = normalizelogs([m.logprobability(data.iloc[i]) + log(p) for m,p in zip(self.models,self.probs)],
                                  weights[i] if weights is not False else 1.0)
                for k in range(knum):
                    ss[k][i] = v[k]
        iter = 0
        while True:
            counts = [prior.alpha/knum + sum(ss[k]) for k in range(knum)]
            s = sum(counts)
            for k in range(knum):
                self.models[k].estimate(data, ss[k], prior.prior)
                self.probs[k] = counts[k]/s
            diff = 0.0
            mdiff = 0.0
            for i in range(dnum):
                v = normalizelogs([m.logprobability(data.iloc[i]) + log(p) for m,p in zip(self.models,self.probs)],
                                  weights[i] if weights is not False else 1.0)
                for k in range(knum):
                    delta = abs(v[k] - ss[k][i])
                    diff += delta
                    mdiff = max(mdiff, delta)
                    ss[k][i] = v[k]
            iter += 1
            print(iter, diff, mdiff)
            if mdiff < em_min_diff or iter >= em_max_iter:
                break
        self.count = sum([m.count for m in self.models])

    def estimate_init(self, prior):
        # note that this is not a good idea before estimate_incr here, use estimate on a seed data set instead.
        # But for estiate_incr_tandem it may be useful to initiate the tandem model.
        knum = len(self.models)
        self.count = prior.alpha
        for k in range(knum):
            self.models[k].estimate_init(prior.prior)
            self.probs[k] = 1/knum
        self.initialized = True

    def estimate_incr(self, sample, weight):
        if len(self.models) == 1:
            self.models[0].estimate_incr(sample, weight)
        else:
            prw = 1.0/max(self.count, 1000)
            v = normalizelogsprior([m.logprobability(sample) + log(p) for m,p in zip(self.models,self.probs)],
                                   weight, prw)
            for i in range(len(self.models)):
                self.models[i].estimate_incr(sample, v[i])
        self.count += weight
        if self.count > 0.0:
            for i in range(len(self.models)):
                self.probs[i] = self.models[i].count/self.count

    def estimate_incr_tandem(self, mod, sample, weight):
        if len(self.models)==1:
            mod.models[0].estimate_incr(sample, weight)
        elif not self.initialized:
            i = randint(0, len(self.models)-1)
            mod.models[i].estimate_incr(sample, weight)
        else:
            prw = 1.0/max(self.count, 1000)
            v = normalizelogsprior([m.logprobability(sample) + log(p) for m,p in zip(self.models,self.probs)],
                                   weight, prw)
            for i in range(len(self.models)):
                mod.models[i].estimate_incr(sample, v[i])
        mod.count += weight
        if mod.count > 0.0:
            for i in range(len(self.models)):
                mod.probs[i] = mod.models[i].count/mod.count

    def init_ss(self, data, weights):
        dnum = len(data)
        knum = len(self.models)
        ss = [[0.0 for i in range(dnum)] for j in range(knum)]
        if dnum < knum*em_init_size: # Doesnt consider weights...
            for i in range(dnum):
                ss[randint(0,knum-1)][i] = 1.0 if weights is False else weights[i]
        elif weights is not False:
            rng = list(range(dnum))
            for j in range(knum*em_init_size):
                i = choices(rng, weights)
                k = randint(0,knum-1)
                if k >= knum or i >= dnum:
                    print("k=%d, i=%d" % (k,i))
                else:
                    ss[k][i] = weights[i]
        else:
            for j in range(knum*64):
                i = randint(0,dnum-1)
                k = randint(0,knum-1)
                if k >= knum or i >= dnum:
                    print("k=%d, i=%d" % (k,i))
                else:
                    ss[k][i] = 1.0
        return ss
        
    def get_ss(self, data, weights):
        dnum = len(data)
        knum = len(self.models)
        ss = [[0.0 for i in range(dnum)] for j in range(knum)]
        for i in range(dnum):
            v = normalizelogs([m.logprobability(data.iloc[i]) + log(p) for m,p in zip(self.models,self.probs)],
                              weights[i] if weights is not False else 1.0)
            for k in range(knum):
                ss[k][i] = v[k]
        return ss
        
    def estimate_ss(self, data, weights, prior, ss, maxiter, mindiff):
        dnum = len(data)
        knum = len(self.models)
        iter = 0
        while True:
            counts = [prior.alpha/knum + sum(ss[k]) for k in range(knum)]
            s = sum(counts)
            for k in range(knum):
                self.models[k].estimate(data, ss[k], prior.prior)
                self.probs[k] = counts[k]/s
            diff = 0.0
            mdiff = 0.0
            for i in range(dnum):
                v = normalizelogs([m.logprobability(data.iloc[i]) + log(p) for m,p in zip(self.models,self.probs)],
                                  weights[i] if weights is not False else 1.0)
                for k in range(knum):
                    delta = abs(v[k] - ss[k][i])
                    diff += delta
                    mdiff = max(mdiff, delta)
                    ss[k][i] = v[k]
            iter += 1
            print(iter, diff, mdiff)
            if mdiff < mindiff or iter >= maxiter:
                break
        return ss

    def estimate_ss_sample(self, data0, weights, prior, ss0, samplesize, maxiter, mindiff):
        dnum = len(data0)
        knum = len(self.models)
        ss = [[0.0 for i in range(samplesize)] for j in range(knum)]
        sst = [[0.0 for i in range(samplesize)] for j in range(knum)]
        rst = sorted(sample(range(dnum), samplesize))
        iter = 0
        while True:
            # slumpa ut submängd
            #if iter % 10 == 0:
            rs = sorted(sample(range(dnum), samplesize))
            data = data0[[i in rs for i in range(dnum)]]
            # beräkna ss för denna
            if iter==0:
                for i in range(samplesize):
                    for k in range(knum):
                        ss[k][i] = ss0[k][rs[i]]
            else:
                for i in range(samplesize):
                    v = normalizelogs([m.logprobability(data.iloc[i]) + log(p) for m,p in zip(self.models,self.probs)],
                                      weights[rs[i]] if weights is not False else 1.0)
                    for k in range(knum):
                        ss[k][i] = v[k]
            counts = [prior.alpha/knum + sum(ss[k]) for k in range(knum)]
            s = sum(counts)
            for k in range(knum):
                self.models[k].estimate(data, ss[k], prior.prior)
                self.probs[k] = (counts[k] + 1)/(s + knum)
            diff = 0.0
            mdiff = 0.0
            for i in range(samplesize):
                v = normalizelogs([m.logprobability(data0.iloc[rst[i]]) + log(p) for m,p in zip(self.models,self.probs)],
                                  weights[rst[i]] if weights is not False else 1.0)
                for k in range(knum):
                    delta = abs(v[k] - sst[k][i])
                    diff += delta
                    mdiff = max(mdiff, delta)
                    sst[k][i] = v[k]
            iter += 1
            print(iter, len(data), diff, mdiff)
            if mdiff < mindiff or iter >= maxiter:
                break
        return self.get_ss(data0, weights)

    def split(self, kind, ss):
        self.probs.insert(kind+1, 0.0)
        self.models.insert(kind+1, self.models[kind].copy_struct())
        s = np.zeros(len(ss[kind]))
        for i in range(len(s)):
            if ss[kind][i]>0.0 and random()>0.5:
                s[i] = ss[kind][i]
                ss[kind][i] = 0.0
        ss.insert(kind+1, s)
        return ss

    def remove(self, kind, ss):
        del ss[kind]
        del self.models[kind]
        del self.probs[kind]
        return ss

    def generate(self, sample):
        thres = random()
        k = len(self.models) - 1
        for i in range(len(self.models)):
            if self.probs[i]>thres:
                k = i
                break
            else:
                thres -= self.probs[i]
        self.models[k].generate(sample)
        return sample

    def probability(self, sample):
        return sum(map(lambda m,p: p*m.probability(sample), self.models, self.probs))

    def logprobability(self, sample):
        lst = list(map(lambda m,p: log(p) + m.logprobability(sample) if p>0.0 else -np.inf, self.models, self.probs))
        mx = max(lst)
        sm = sum(map(lambda x: exp(x-mx), lst))
        return mx + log(sm)

    def logmodelprob(self, prior):
        sum = 0.0
        for i in range(len(self.models)):
            sum += self.models[i].logmodelprob(prior.prior) + log(self.probs[i])*prior.alpha
        return sum

    def margin(self, mlst):
        return MixtureModel([m.margin(mlst) for m in self.models], self.probs)

    def predict(self, sample):
        flst = [f for f in self.feet if not np.isnan(sample[f])]
        probs = normalizelogs(map(lambda m,p: m.margin(flst).logprobability(sample) + log(p), self.models, self.probs), 1.0)
        return MixtureModel([m.predict(sample) for m in self.models], probs)

    def anomaly(self, sample):
        # pick the anomaly of the least anomalous cluster, adjusted for cluster prob
        mina = np.inf
        for ind in range(len(self.probs)):
            p = self.probs[ind]
            cump = sum([pp for pp in self.probs if pp <= p])
            ano = self.models[ind].anomaly(sample) - log(cump)
            if ano < mina:
                mina = ano
        return mina

    def multiply(self, modlst, cntlst):
        pass ##
    
    def average(self, modlst):
        submodels = [m.models for m in modlst]
        nmodlst = [submodels[0][i].average([submodels[j][i] for j in range(len(modlst))]) for i in range(len(submodels[0]))]
        wsum = sum([m.count for m in modlst])
        probs = cubeweightedsum([m.probs for m in modlst], [m.count/wsum for m in modlst])
        return MixtureModel(nmodlst, probs)

    def prior(self, alpha, data):
        return MixturePrior(self, alpha, self.form, data)

    def copy_struct(self):
        return MixtureModel(self.models[0], len(self.models))


class MixturePrior(Prior):
    def __init__(self, model, alpha, form, data):
        Prior.__init__(self, alpha)
        self.prior = model.models[0].prior(alpha/len(model.models), data)
#        self.prior = model.models[0].prior(alpha, data)

#----------------------------------------------------------
# ClassMixtureModel
#----------------------------------------------------------

class ClassMixtureModel(Model):
    def __init__(self, idx, form, mod, prob = False):
        if type(mod)==list and type(prob)==list:
            Model.__init__(self, [idx] + mod[0].feet, form)
            num = len(mod)
            self.models = mod
            self.probs = prob
            self.count = sum([m.count for m in mod])
        else:
            Model.__init__(self, [idx] + mod.feet, form)
            num = form[idx].num
            self.models = [mod.copy_struct() for i in range(num)]
            self.probs = [1.0/num for i in range(num)]

    def estimate(self, data, weights, prior):
        # läs ut komponent-tillhörigheter
        # träna komponenterna med dessa
        dnum = len(data)
        knum = len(self.models)
        ss = [[0.0 for i in range(dnum)] for j in range(knum)]
        for i in range(dnum):
            ss[data[self.feet[0]][i]][i] = 1.0 if weights is False else weights[i]
        for k in range(knum):
            self.models[k].estimate(data, ss[k], prior)
        self.count = sum([m.count for m in self.models])

    def estimate_init(self, prior):
        knum = len(self.models)
        self.count = prior.alpha
        for k in range(knum):
            self.models[k].estimate_init(prior.prior)
            self.probs[k] = 1/knum

    def estimate_incr(self, sample, weight):
        v = normalizelogs([m.logprobability(sample) + log(p) for m,p in zip(self.models,self.probs)],
                          weight)
        for i in range(len(self.models)):
            self.models[i].estimate_incr(sample, v[i])

    def generate(self, sample):
        thres = random()
        k = len(self.models) - 1
        for i in range(len(self.models)):
            if self.probs[i]>thres:
                k = i
                break
            else:
                thres -= self.probs[i]
        sample[self.feet[0]] = k
        self.models[k].generate(sample)
        return sample

    def probability(self, sample):
        k = sample[self.feet[0]]
        return self.probs[k] * self.models[k].probability(sample)

    def logprobability(self, sample):
        k = sample[self.feet[0]]
        return log(self.probs[k]) + self.models[k].logprobability(sample)

    def logmodelprob(self, prior):
        sum = 0.0
        for i in range(len(self.models)):
            sum += self.models[i].logmodelprob(prior.prior) + log(self.probs[i])*prior.alpha
        return sum

    def margin(self, mlst):
        if self.feet[0] in mlst:
            return ClassMixtureModel(self.feet[0], self.form, [m.margin(mlst) for m in self.models], self.probs)
        else:
            return MixtureModel([m.margin(mlst) for m in self.models], self.probs)

    def predict(self, sample):
        if np.isnan(sample[self.feet[0]]):
            flst = [f for f in self.feet if not np.isnan(sample[f])]
            probs = normalizelogs(map(lambda m,p: m.margin(flst).logprobability(sample) + log(p), self.models, self.probs), 1.0)
            return ClassMixtureModel(self.feet[0], self.form, [m.predict(sample) for m in self.models], probs)
        else:
            ind = sample[self.feet[0]]
            return MixtureModel([self.models[ind].predict(sample)], [1.0])

    def anomaly(self, sample):
        if np.isnan(sample[self.feet[0]]):
            mina = np.inf
            for ind in range(len(self.probs)):
                p = self.prob[ind]
                cump = sum([pp for pp in self.prob if pp <= p])
                ano = self.models[ind].anomaly(sample) - log(cump)
                if ano < mina:
                    mina = ano
            return mina
        else:
            ind = sample[self.feet[0]]
            p = self.prob[ind]
            cump = sum([pp for pp in self.prob if pp <= p])
            return self.models[ind].anomaly(sample) - log(cump)

    def multiply(self, modlst, cntlst):
        pass ##
    
    def average(self, modlst):
        submodels = [m.models for m in modlst]
        nmodlst = [submodels[0][i].average([submodels[j][i] for j in range(len(modlst))]) for i in range(len(submodels[0]))]
        wsum = sum([m.count for m in modlst])
        probs = cubeweightedsum([m.probs for m in modlst], [m.count/wsum for m in modlst])
        return ClassMixtureModel(self.feet[0], self.form, nmodlst, probs)

    def prior(self, alpha, data):
        return MixturePrior(self, alpha, self.form, data)

    def copy_struct(self):
        return ClassMixtureModel(self.feet[0], self.form, self.models[0])


#----------------------------------------------------------
# GraphModel
#----------------------------------------------------------

class GraphModel(Model):
    def __init__(self, models):
        Model.__init__(self, flattenuniq([m.feet for m in models]), models[0].form if models is not [] else {})
        self.models = models
        self.struct = [self.submodinds(m) for m in self.models]

    def submodinds(self, mod):
        res = []
        for i in range(len(self.models)):
            if issubset(self.models[i].feet, mod.feet):
                res.append(i)
        return res

    def estimate(self, data, weights, prior):
        for i in range(len(self.models)):
            self.models[i].estimate(data, weights, prior.priors[i])
        self.count = max([m.count for m in self.models])

    def estimate_init(self, prior):
        for i in range(len(self.models)):
            self.models[i].estimate_init(prior.priors[i])
        self.count = prior.alpha

    def estimate_incr(self, sample, weight):
        for i in range(len(self.models)):
            self.models[i].estimate_incr(sample, weight)
        self.count += weight

    def generate(self, sample):
        pass

    def modcount(self, cnt, i, c):
        cnt[i] += c
        for j in self.struct[i]:
            self.modcount(cnt, j, -c)

    def probability(self, sample):
        return exp(self.logprobability(sample))

    def logprobability(self, sample):
        cnt = [0 for i in range(len(self.models))]
        prob = [0.0 for i in range(len(self.models))]
        for i in range(len(self.models)):
            prob[i] = self.models[i].logprobability(sample)
            self.modcount(cnt, i, 1)
        return sum(map(lambda p,c: p*c, prob, cnt))

    def logmodelprob(self, prior):
        sum = 0.0
        for i in range(len(self.models)):
            sum += self.models[i].logmodelprob(prior.priors[i])
        return sum

    def margin(self, mlst):
#        nmods = []
#        nstr = []
#        for i in range(len(self.models)):
#            if subeqset**(self.struct[i]**, flst**):
#                nmods.append(self.models[i])
#                nstr.append()
        pass ##

    def predict(self, sample):
        pass ##

    def anomaly(self, sample):
        pass ##

    def multiply(self, modlst, cntlst):
        pass ##
    
    def average(self, modlst):
        submodels = [m.models for m in modlst]
        nmodlst = [submodels[0][i].average([submodels[j][i] for j in range(len(modlst))]) for i in range(len(submodels[0]))]
        return GraphModel(nmodlst)
        
    def prior(self, alpha, data):
        return GraphPrior(self, alpha, self.form, data)

    def copy_struct(self):
        return GraphModel([m.copy_struct() for m in self.models])


class GraphPrior(Prior):
    def __init__(self, model, alpha, form, data):
        Prior.__init__(self, alpha)
        self.priors = [m.prior(alpha, data) for m in model.models]


#------------------------------------------

def modellikelihood(mod, data, weights):
    llsum = 0.0
    wsum = 0.0
    if weights is not False:
        for i in range(len(data)):
            llsum += mod.logprobability(data.iloc[i]) * weights[i]
        wsum = sum(weights)
    else:
        for i in range(len(data)):
            llsum += mod.logprobability(data.iloc[i])
        wsum = len(data)
    return llsum/wsum if wsum > 0.0 else 0.0

def clustersimilarities(s):
    var = [sqrt(np.matmul(s[i],s[i])) for i in range(len(s))]
    cov = {}
    for i in range(len(s)-1):
        for j in range(i+1, len(s)):
            cov[(i,j)] = np.matmul(s[i],s[j])/(var[i]*var[j])
    return cov

def clustersimilarities2(s1, s2):
    var1 = [sqrt(np.matmul(s1[i],s1[i])) for i in range(len(s1))]
    var2 = [sqrt(np.matmul(s2[j],s2[j])) for j in range(len(s2))]
    cov = {}
    for i in range(len(s1)):
        for j in range(len(s2)):
            cov[(i,j)] = np.matmul(s1[i],s2[j])/(var1[i]*var2[j])
    return cov

def selectind(acc):
    num=len(acc)
    r=random()*acc[-1]
    ind1 = 0
    ind2 = num-1
    while ind1 < ind2:
        ind = floor((ind1+ind2)/2)
        if acc[ind] > r:
            ind2 = ind
        else:
            ind1 = ind+1
    return ind1

def gammatest(comp, data, weights, num):
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
    oldind = selectind(acc)
    ind = oldind
    wsum = 0
    for i in range(num):
        while ind == oldind:
            ind = selectind(acc)
        dist = sqvec(np.matmul(comp.ilower, np.subtract(data.iloc[ind], data.iloc[oldind])))
        w = weights[oldind] # Because weights[ind] is already accounted for in selection
        dists[i] = (dist, w)
        wsum += w
        oldind = ind
    dists.sort(key = lambda pr:pr[0])
    mnval = 0
    mnind = 0
    w = 0
    for i in range(num):
        (d, w0) = dists[i]
        w += w0
        p = gammainc(dim/2, d/4)
        # print((wsum, w, p))
        val = log(wsum*p/w)*w + log(wsum*(1-p)/(wsum-w))*(wsum-w) if p > 0 and p < 1 and w > 0 and w < wsum else 0
        dists[i] = (d, w, p, p*wsum - w, val) 
        if val<mnval and p<0.90 and p>0.05 and p*wsum > w:
            mnval = val
            mnind = i
    (d, w,z1,z2,z3) = dists[mnind]
    p = gammainc(dim/2, d/4)
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
    return (sig, dists)

