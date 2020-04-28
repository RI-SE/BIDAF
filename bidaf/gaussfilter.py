from math import *

def gaussf(x, s):
    return exp(-x*x/(2*s*s))

class acc():
    def __init__(self, tt, s):
        self.tt = tt
        self.s = s
        self.sn = 0.0
        self.sx = 0.0
        self.st = 0.0
        self.sxt = 0.0
        self.sxx = 0.0
        self.stt = 0.0
        self.lastt = None

    def add(self, ti, xi):
        g = gaussf(ti-self.tt, self.s)
        self.lastt = ti
        self.sn += g
        self.sx += g*xi
        self.st += g*(ti-self.tt)
        self.sxt += g*xi*(ti-self.tt)
        self.sxx += g*xi*xi
        self.stt += g*(ti-self.tt)*(ti-self.tt)

    def calc(self):
        if self.sn==0.0:
            return [0.0, 0.0, 0.0, 0.0]
        mm = self.sx/self.sn
        kk = (self.sxt - mm*self.st)/(self.sn*self.s*self.s)
        v0 = self.sxx/self.sn - mm*mm
        v1 = v0 + (self.stt/self.sn - 2.0*self.s*self.s)*kk*kk
        return [mm, kk, sqrt(max(0.0, v0)), sqrt(max(0.0, v1))]

class GaussFilter:
    def __init__(self, func, timeattr, entityattr, valueattrs, interval, scales, slope=False, variance=False, residual=False):
        # accum är en dictionary (entiteter) av lista (tidpunkterna)
        # av lista med en central-tid, senast-tid och en dictionary
        # (variablerna) av lista (skalorna) av accumulatorer.
        # deliver är en funktion som tar ett resultat-dict och anropas
        # varje interval
        self.deliver = func
        self.timeattr = timeattr
        self.entityattr = entityattr
        self.valueattrs = valueattrs
        self.interval = interval
        self.slope = slope
        self.variance = variance
        self.residual = residual
        self.scales = scales
        self.accum = {}
        self.horizon = max(scales)*2
        self.lasttime = None

    def nexttime(self, tm):
        return ceil(tm/self.interval)*self.interval

    def resultkeys(self):
        if self.entityattr:
            res = [self.timeattr, self.entityattr]
        else:
            res = [self.timeattr]
        for key in self.valueattrs:
            for s in self.scales:
                res.append(key + '_mean' + str(s))
                if self.slope:
                    res.append(key + '_slope' + str(s))
                if self.variance:
                    res.append(key + '_stdev' + str(s))
        return res

    def resultdict(self, ent, t, vardic):
        if self.entityattr:
            resdic = {self.timeattr: t, self.entityattr: ent}
        else:
            resdic = {self.timeattr: t}
        for key in vardic:
            for a in vardic[key]:
                vals = a.calc()
                tmpkey = key + '_mean' + str(a.s)
                resdic[tmpkey] = vals[0]
                if self.slope:
                    tmpkey = key + '_slope' + str(a.s)
                    resdic[tmpkey] = vals[1]
                if self.variance:
                    tmpkey = key + '_stdev' + str(a.s)
                    resdic[tmpkey] = vals[3] if self.slope else vals[2]
            # residualer
            if self.residual:
                tmplst = sorted(self.scales, key=lambda x: x)
                for s1,s2 in zip(tmplst[:-1],tmplst[1:]):
                    tmpkey1 = key + '_mean' + str(s1)
                    tmpkey2 = key + '_mean' + str(s2)
                    resdic[tmpkey1] -= resdic[tmpkey2]
                    if self.slope:
                        tmpkey1 = key + '_slope' + str(s1)
                        tmpkey2 = key + '_slope' + str(s2)
                        resdic[tmpkey1] -= resdic[tmpkey2]
                if self.variance:
                    tmplst = sorted(self.scales, key=lambda x: -x)
                    for s1,s2 in zip(tmplst[:-1],tmplst[1:]):
                        tmpkey1 = key + '_stdev' + str(s1)
                        tmpkey2 = key + '_stdev' + str(s2)
                        resdic[tmpkey1] = sqrt(max(0.0, resdic[tmpkey1]*resdic[tmpkey1] - resdic[tmpkey2]*resdic[tmpkey2]))
        return resdic

    def handle(self, dic):
        tm = dic[self.timeattr]
        # samla ihop de som ska levereras, och leverera i tidsordning
        if self.lasttime is not None and tm < self.lasttime - self.horizon:
            fromtime = inf
        else:
            fromtime = self.nexttime(tm - self.horizon)
        createtime = self.nexttime(tm)
        self.lasttime = tm
        dlst = []
        for ent in self.accum:
            tlst = self.accum[ent]
            while len(tlst) > 0 and tlst[0][0] < fromtime:
                tmp = tlst.pop(0)
                if tmp[1] > tmp[0]:
                    dlst.append(self.resultdict(ent, tmp[0], tmp[2]))
        dlst.sort(key=lambda ele: ele[self.timeattr])
        for ele in dlst:
            self.deliver(ele)
        ent = dic[self.entityattr] if self.entityattr else True
        if not ent in self.accum:
            self.accum[ent] = []
        tlst = self.accum[ent]
        # se till att tidpunkter finns
        if len(tlst)==0 or tlst[-1][0] < createtime:
            t = createtime
        else:
            t = tlst[-1][0] + self.interval
        while t < tm + self.horizon:
            tlst.append([t, tm, {}])
            t += self.interval
        # registrera värden
        for key in self.valueattrs:
            x =  dic[key]
            for tmp in tlst:
                vardic = tmp[2]
                tmp[1] = tm
                if not key in vardic:
                    vardic[key] = [acc(tmp[0], s) for s in self.scales]
                for a in vardic[key]:
                    a.add(tm, x)
                    
    def reset(self):
        # samla ihop de som ska levereras, och leverera i tidsordning
        dlst = []
        for ent in self.accum:
            tlst = self.accum[ent]
            while len(tlst) > 0 and tlst[0][0] < fromtime:
                tmp = tlst.pop(0)
                if tmp[1] > tmp[0]:
                    dlst.append(self.resultdict(ent, tmp[0], tmp[2]))
        dlst.sort(key=lambda ele: ele[self.timeattr])
        for ele in dlst:
            self.deliver(ele)
