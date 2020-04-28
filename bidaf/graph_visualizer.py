import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from random import *

from Color import *

dircol = 0.02
causcol = 0.84
simicol = 0.18

def tupleadd(x, y, mul=1):
    return (x[0]+mul*y[0], x[1]+mul*y[1])

def tupledot(x, y):
    return x[0]*y[0] + x[1]*y[1]

def subsets(lst, ord):
    if ord==0:
        return [[]]
    elif  ord > len(lst):
        return []
    else:
        rest = subsets(lst[1:], ord-1)
        return list(map(lambda l:[lst[0]]+l, rest)) + subsets(lst[1:],ord)

def discrete(n):
    return int(random()*n)

class GraphNode():
    def __init__(self, win, name, pos, rad, prob):
        self.win = win
        self.name = name
        self.pos = pos
        self.rad = rad
        self.prob = prob
        self.links = []
        self.hlinks = []
        self.histgr = False
        self.selected = False
        xsc = win.xscale()
        ysc = win.yscale()
        p1 = win.trans(pos)
        p2 = tupleadd(p1, (xsc, -ysc), 1.0)
        p3 = tupleadd(p1, (xsc, -ysc), -1.0)
        gr1 = hsl_color(0, 0, -0.2)
        gr2 = hsl_color(0, 0, -0.5)
        gr3 = hsl_color(0, 0, 0.3)
        xdiam = xsc*rad*2
        ydiam = ysc*rad*2
        c1 = mpl.patches.Ellipse(p1, xdiam, ydiam, linewidth=6*self.win.dotwidth, edgecolor=gr1, facecolor='white')
        c2 = mpl.patches.Ellipse(p2, xdiam, ydiam, linewidth=3*self.win.dotwidth, edgecolor=gr2, fill=False)
        c3 = mpl.patches.Ellipse(p3, xdiam, ydiam, linewidth=3*self.win.dotwidth, edgecolor=gr3, fill=False)
        c4 = mpl.patches.Ellipse(p1, xdiam-2*xsc, ydiam-2*ysc, linewidth=2*self.win.dotwidth, edgecolor=gr1, fill=False)
        #c4 = mpl.patches.Ellipse(p1, xdiam+2*xsc, ydiam+2*ysc, linewidth=2*self.win.dotwidth, edgecolor=gr1, fill=False)
        t1 = plt.Text(p1[0], p1[1], name, ha='center', fontsize=self.win.fontsz1*self.win.dotwidth)
        t2 = plt.Text(p1[0], p1[1]-self.win.fontsz1*ysc, format(prob, ".2f") if prob else "", ha='center', fontsize=self.win.fontsz0*self.win.dotwidth)
        self.nametext = t1
        self.probtext = t2
        self.circles = [c1, c2, c3, c4]
        self.extras = []
        for obj in self.circles + [t1, t2]:
            self.win.fig.add_artist(obj)
        self.win.nodes[name] = self

    def set_color(self, col):
        self.color = col
        self.circles[0].set_facecolor(col)

    def set_selected(self, val):
        self.selected = val
        if val:
            self.circles[3].width = (2*self.rad + 2)*self.win.xscale()
            self.circles[3].height = (2*self.rad + 2)*self.win.yscale()
        else:
            self.circles[3].width = (2*self.rad - 2)*self.win.xscale()
            self.circles[3].height = (2*self.rad - 2)*self.win.yscale()

    def set_prob(self, prob):
        self.prob = prob
        self.probtext.set_text(format(prob, ".2f") if prob is not False else "")
        #plt.draw()

    def set_distr(self, hist, num):
        if self.histgr is False:
            self.histgr = Histogram(self.win, num, self.circles[0].zorder, symmetric=(hist[0][0]<0.0))
        self.histgr.set(hist, 'Fixed')
        self.histgr.move(tupleadd(self.circles[0].get_center(), (0.6*self.win.radius*self.win.xscale(), 0.6*self.win.radius*self.win.yscale())))
        self.histgr.show()
        #plt.draw()

    def move(self, pos):
        delta = tupleadd(pos, self.pos, -1)
        delta = (delta[0]*self.win.xscale(), delta[1]*self.win.yscale())
        self.pos = pos
        for c in self.circles:
            c.set_center(tupleadd(c.get_center(), delta))
        for t in [self.nametext, self.probtext]:
            t.set_position(tupleadd(t.get_position(), delta))
        for l in self.links:
            l.updatelinepos()
        for l in self.hlinks:
            l.updatepathpos()
        if self.extras is not []:
            for e in self.extras:
                e.set_xdata(tupleadd(e.get_xdata(), (delta[0], delta[0])))
                e.set_ydata(tupleadd(e.get_ydata(), (delta[1], delta[1])))
        if self.histgr is not False:
            self.histgr.move(tupleadd(self.circles[0].get_center(), (0.6*self.win.radius*self.win.xscale(), 0.6*self.win.radius*self.win.yscale())))
        #plt.draw()

    def remove(self):
        for l in self.links.copy():
            l.remove()
        for c in self.circles:
            c.remove()
        for t in [self.nametext, self.probtext]:
            t.remove()
        if self.extras is not []:
            for e in self.extras:
                e.remove()
        del self.win.nodes[self.name]
        if self.histgr is not False:
            self.histgr.remove()
        #plt.draw()

    def todict(self):
        return {'name': self.name, 'pos': self.pos, 'prob': self.prob}


class GraphLink():
    def __init__(self, win, name1, name2, tp, strn):
        self.win = win
        self.fr = name1
        self.to = name2
        self.type = ""
        self.strength = 0.0
        self.istrength = (0.0, 0.0, 0.0)
        self.dstrength = (0.0, 0.0, 0.0)
        self.sstrength = 0.0
        n1 = win.nodes[name1]
        n2 = win.nodes[name2]
        n1.links.append(self)
        if not n1 == n2:
            n2.links.append(self)
        if name1 == name2:
            xdiam = win.xscale()*win.radius*2
            ydiam = win.yscale()*win.radius*2
            self.line = mpl.patches.Ellipse((0,0), xdiam, ydiam, linewidth=win.thickness*self.win.dotwidth, fill=False)
        else:
            self.line = plt.Line2D((0,0), (0,0), linewidth=win.thickness*self.win.dotwidth)
        self.arrow = [plt.Line2D((0,0), (0,0), linewidth=win.thickness*self.win.dotwidth),
                      plt.Line2D((0,0), (0,0), linewidth=win.thickness*self.win.dotwidth)]
        zo = min(n1.circles[0].zorder, n2.circles[0].zorder) -2
        self.line.zorder = zo
        self.arrow[0].zorder = zo
        self.arrow[1].zorder = zo
        self.text = [plt.Text(0, 0, "", fontsize=self.win.fontsz0*self.win.dotwidth),
                     plt.Text(0, 0, "", fontsize=self.win.fontsz0*self.win.dotwidth)]
        self.text[0].zorder = zo
        self.text[1].zorder = zo
        self.histgr = False
        self.modify(name1, name2, tp, strn)
        tup = (self.fr, self.to) if self.fr < self.to else (self.to, self.fr)
        self.win.links[tup] = self
        win.fig.add_artist(self.line)
        win.fig.add_artist(self.arrow[0])
        win.fig.add_artist(self.arrow[1])
        win.fig.add_artist(self.text[0])
        win.fig.add_artist(self.text[1])

    def remove(self):
        n1 = self.win.nodes[self.fr]
        n2 = self.win.nodes[self.to]
        n1.links.remove(self)
        if not n1 == n2:
            n2.links.remove(self)
        tup = (self.fr, self.to) if self.fr < self.to else (self.to, self.fr)
        del self.win.links[tup]
        for o in [self.line] + self.arrow + self.text:
            o.remove()
        if self.histgr is not False:
            self.histgr.remove()
        #plt.draw()

    def modify(self, name1, name2, tp, strn):
        if ((tp == "Causal" or self.type == "Causal" or self.type == "") and
            (self.fr != name1 or self.to != name2 or self.type != tp)):
            dirtylinepos = True
        else:
            dirtylinepos = False
        if (tp in ["Indirect","Highorder","None"]):
            if tp == "Indirect":
                self.istrength = strn
                if abs(self.istrength[0])+0.25 < abs(self.sstrength):
                    tp = "Highorder"
            elif tp == "Highorder":
                self.sstrength = strn
                if not self.type in ["Indirect","None",""] or abs(self.istrength[0])+0.25 >= abs(self.sstrength):
                    tp = self.type
                    name1 = self.fr
                    name2 = self.to
            else:
                self.istrength = 0.0
                if 0.25 < abs(self.sstrength):
                    tp = "Highorder"
        elif (tp == "Direct" or tp == "Causal"):
            self.dstrength = strn
        elif tp == "Fixed":
            self.strength = strn
        self.fr = name1
        self.to = name2
        self.type = tp
        if dirtylinepos:
            self.updatelinepos()
        if tp == "Direct":
            self.text[0].set_text(self.win.format_strength(self.istrength))
            self.text[1].set_text(self.win.format_strength(self.dstrength))
            self.text[1].set_color(hsl_color(dircol, 1.0, -0.2))
            if self.win.convertmi:
                lcol = hsl_color(dircol, 1.0, 0.8 - np.sqrt(min(1.0, 2.0*abs(self.dstrength[0]))))
            else:
                lcol = hsl_color(dircol, 1.0, 0.8 - abs(self.dstrength[0]))
            if self.histgr is not False:
                self.histgr.set(None, tp)
                self.histgr.show()
        elif tp == "Causal":
            self.text[0].set_text(self.win.format_strength(self.istrength))
            self.text[1].set_text(self.win.format_strength(self.dstrength))
            self.text[1].set_color(hsl_color(causcol, 1.0, -0.2))
            if self.win.convertmi:
                lcol = hsl_color(causcol, 1.0, 0.8 - np.sqrt(min(1.0, 2.0*abs(self.dstrength[0]))))
            else:
                lcol = hsl_color(causcol, 1.0, 0.8 - abs(self.dstrength[0]))
            if self.histgr is not False:
                self.histgr.set(None, tp)
                self.histgr.show()
        elif tp == "Indirect":
            self.text[0].set_text("")
            self.text[1].set_text(self.win.format_strength(self.istrength))
            self.text[1].set_color('black')
            if self.win.convertmi:
                lcol = hsl_color(0.0, 0.0, 0.8 - 1.8*np.sqrt(min(1.0, 2.0*abs(self.istrength[0]))))
            else:
                lcol = hsl_color(0.0, 0.0, 0.8 - 1.8*abs(self.istrength[0]))
            if self.histgr is not False:
                self.histgr.set(None, tp)
                self.histgr.show()
        elif tp == "Highorder":
            self.text[0].set_text(self.win.format_strength(self.istrength))
            self.text[1].set_text(self.win.format_strength(self.sstrength))
            self.text[1].set_color(hsl_color(simicol, 0.8, -0.2))
            if self.win.convertmi:
                lcol = hsl_color(simicol, 0.8, 0.8 - 1.8*np.sqrt(min(1.0, 2.0*abs(self.sstrength))))
            else:
                lcol = hsl_color(simicol, 0.8, 0.8 - 1.8*abs(self.sstrength))
            if self.histgr is not False:
                self.histgr.set(None, tp)
                self.histgr.show()
        elif tp == "Fixed":
            self.text[0].set_text("")
            self.text[1].set_text(self.win.format_strength(self.strength))
            self.text[1].set_color('black')
            lcol = hsl_color(0.0, 0.0, -1.0)
            if self.histgr is not False:
                self.histgr.show()
        else:
            self.strength = 0.0
            self.istrength = (0.0, 0.0, 0.0)
            self.dstrength = (0.0, 0.0, 0.0)
            self.text[0].set_text("")
            self.text[1].set_text("")
            lcol = (0,0,0,0)
            if self.histgr is not False:
                self.histgr.hide()
        if self.fr == self.to:
            self.line.set_edgecolor(lcol)
        else:
            self.line.set_color(lcol)
        self.arrow[0].set_color(lcol)
        self.arrow[1].set_color(lcol)
        #plt.draw()

    def set_distr(self, hist, num):
        if self.histgr is False:
            self.histgr = Histogram(self.win, num, self.line.zorder)
        self.histgr.set(hist, self.type)
        self.updatelinepos()
        self.histgr.show()
        #plt.draw()

    def updatelinepos(self):
        p1 = self.win.trans(self.win.nodes[self.fr].pos)
        p2 = self.win.trans(self.win.nodes[self.to].pos)
        if self.fr == self.to:
            self.line.set_center(tupleadd(p1, (0, self.win.radius*self.win.yscale())))
            pc = tupleadd(p1, (self.win.radius*self.win.xscale(), self.win.radius*self.win.yscale()))
            v = (0.1, -1)
        else:
            self.line.set_xdata((p1[0], p2[0]))
            self.line.set_ydata((p1[1], p2[1]))
            pc = ((p1[0] + p2[0])*0.5, (p1[1] + p2[1])*0.5)
            v = (p2[0] - p1[0], p2[1] - p1[1])
        vlen = np.sqrt(tupledot(v, v))
        if vlen == 0.0:
            vlen = 1
        vnormal = (-v[1]/vlen, v[0]/vlen) if v[0] > 0 else (v[1]/vlen, -v[0]/vlen)
        vunit = (v[0]/vlen, v[1]/vlen)
        if self.type == "Causal" or self.type == "Fixed":
            a1 = tupleadd(pc, tupleadd(vnormal, vunit, -1), self.win.arrowsize*self.win.xscale())
            a2 = tupleadd(pc, tupleadd(vnormal, vunit, 1), -self.win.arrowsize*self.win.xscale())
        else:
            a1 = pc
            a2 = pc
        self.arrow[0].set_xdata((pc[0], a1[0]))
        self.arrow[0].set_ydata((pc[1], a1[1]))
        self.arrow[1].set_xdata((pc[0], a2[0]))
        self.arrow[1].set_ydata((pc[1], a2[1]))
        if self.histgr is not False:
            bdim = (54*0.5*self.win.xscale(), 32*0.5*self.win.yscale())
            bdimmarg = (74*0.5*self.win.xscale(), 52*0.5*self.win.yscale())
        else:
            bdim = (32*0.5*self.win.xscale(), 16*0.5*self.win.yscale())
            bdimmarg = (44*0.5*self.win.xscale(), 32*0.5*self.win.yscale())
        bdist = tupledot((abs(vnormal[0]), vnormal[1]), bdimmarg)
        tpos = tupleadd(tupleadd(pc, vnormal, bdist), bdim, -1)
        self.text[0].set_position(tupleadd(tpos, (0, self.win.fontsz0*self.win.yscale())))
        self.text[1].set_position(tpos)
        if self.histgr is not False:
            self.histgr.move(tupleadd(tpos, (0, 2*self.win.fontsz0*self.win.yscale())))

    def isclose(self, ev, tol):
        pxoff = self.win.fig.transFigure.transform((self.win.offx,self.win.offy))
        if self.fr == self.to:
            p1 = tupleadd(self.win.nodes[self.fr].pos, pxoff)
            rad = self.win.radius
            c1 = tupleadd(p1, (0, rad))
            p0 = (ev.x - c1[0], ev.y - c1[1])
            d = np.sqrt(tupledot(p0, p0))
            return abs(d - rad) < tol
        else:
            p1 = tupleadd(self.win.nodes[self.fr].pos, pxoff)
            p2 = tupleadd(self.win.nodes[self.to].pos, pxoff)
            p0 = (ev.x - p1[0], ev.y - p1[1])
            r1 = (p2[0] - p1[0], p2[1] - p1[1])
            r2 = (r1[1], -r1[0])
            l2 = tupledot(r1, r1)
            d1 = tupledot(p0, r1)
            d2 = tupledot(p0, r2)
            return d1 > 0.0 and d1 < l2 and d2*d2 < tol*tol*l2

    def todict(self):
        return {'from': self.fr, 'to': self.to, 'strength': self.strength}


class GraphHyperlink():
    def __init__(self, win, name1, name2, name3, tp, strn):
        self.win = win
        self.name1 = name1
        self.name2 = name2
        self.name3 = name3
        self.type = ""
        self.istrength = (0.0, 0.0, 0.0)
        self.dstrength = (0.0, 0.0, 0.0)
        self.cnum = 20
        self.clst = [np.cos(i*np.pi/(2*self.cnum)) for i in range(self.cnum+1)]
        n1 = win.nodes[name1]
        n2 = win.nodes[name2]
        n3 = win.nodes[name3]
        n1.hlinks.append(self)
        n2.hlinks.append(self)
        n3.hlinks.append(self)
        self.path = mpl.patches.Polygon([[0,0],[0,0],[0,0]], fill=True, closed=True)
        self.text = [plt.Text(0, 0, "", fontsize=self.win.fontsz0*self.win.dotwidth),
                     plt.Text(0, 0, "", fontsize=self.win.fontsz0*self.win.dotwidth)]
        zo = min(n1.circles[0].zorder, n2.circles[0].zorder) -2
        self.path.zorder = zo
        self.text[0].zorder = zo
        self.text[1].zorder = zo
        self.histgr = False
        self.modify(tp, strn)
        self.updatepathpos()
        self.win.hlinks[tuple(sorted([self.name1, self.name2, self.name3]))] = self
        win.fig.add_artist(self.path)
        win.fig.add_artist(self.text[0])
        win.fig.add_artist(self.text[1])

    def modify(self, tp, strn):
        self.type = tp
        if tp == "Direct":
            self.dstrength = strn
            self.text[0].set_text(self.win.format_strength(self.istrength))
            self.text[1].set_text(self.win.format_strength(self.dstrength))
            self.text[1].set_color(hsl_color(dircol, 1.0, -0.2))
            lcol = hsl_color(dircol, 1.0, 0.0) + (0.4,)
            if self.histgr is not False:
                self.histgr.set(None, tp)
                self.histgr.show()
        elif tp == "Indirect":
            self.istrength = strn
            self.text[0].set_text("")
            self.text[1].set_text(self.win.format_strength(self.istrength))
            self.text[1].set_color('black')
            lcol = hsl_color(0.0, 0.0, 0.1) + (0.3,)
            if self.histgr is not False:
                self.histgr.set(None, tp)
                self.histgr.show()
        else:
            self.istrength = 0.0
            self.dstrength = 0.0
            self.text[0].set_text("")
            self.text[1].set_text("")
            lcol = (0,0,0,0)
            if self.histgr is not False:
                self.histgr.hide()
        self.path.set_facecolor(lcol)
        #plt.draw()

    def set_distr(self, hist, num):
        if self.histgr is False:
            self.histgr = Histogram(self.win, num, self.path.zorder)
        self.histgr.set(hist, self.type)
        self.updatepathpos()
        self.histgr.show()
        #plt.draw()

    def updatepathpos(self):
        plst = list(map(lambda n: self.win.trans(self.win.nodes[n].pos),
                        [self.name1, self.name2, self.name3]))
        cent = (0,0)
        for p in plst:
            cent = tupleadd(cent, p, 1/3)
        l1 = [tupleadd(tupleadd(tupleadd(tupleadd(plst[0],plst[1]), cent, -1),
                                tupleadd(cent, plst[0], -1), self.clst[i]),
                       tupleadd(cent, plst[1], -1), self.clst[self.cnum-i])
              for i in range(self.cnum)]
        l2 = [tupleadd(tupleadd(tupleadd(tupleadd(plst[1],plst[2]), cent, -1),
                                tupleadd(cent, plst[1], -1), self.clst[i]),
                       tupleadd(cent, plst[2], -1), self.clst[self.cnum-i])
              for i in range(self.cnum)]
        l3 = [tupleadd(tupleadd(tupleadd(tupleadd(plst[2],plst[0]), cent, -1),
                                tupleadd(cent, plst[2], -1), self.clst[i]),
                       tupleadd(cent, plst[0], -1), self.clst[self.cnum-i])
              for i in range(self.cnum)]
        self.path.set_xy(l1+l3+l2)
        self.text[0].set_position(tupleadd(cent, (-30*self.win.xscale(), -self.win.fontsz0*self.win.yscale())))
        self.text[1].set_position(tupleadd(cent, (-30*self.win.xscale(), -2*self.win.fontsz0*self.win.yscale())))
        if self.histgr is not False:
            self.histgr.move(tupleadd(cent, (-30*self.win.xscale(), 0)))


class Histogram():
    def __init__(self, win, nbins, zorder, symmetric=True):
        self.win = win
        self.num = nbins
        self.width = 2*(nbins//2)+(3.0 if symmetric else 2.0)
        self.height = int(nbins*0.75)+1.0
        self.hidden = True
        self.hist = []
        self.type = 'None'
        self.symmetric = symmetric
        col = (0,0,0,0)
        xscale = self.win.xscale()
        yscale = self.win.yscale()
        self.rect = plt.Rectangle((0,0), xscale*self.width, yscale*self.height, linewidth=self.win.dotwidth, facecolor=col, edgecolor=col, zorder=zorder)
        self.line = plt.Line2D((0,0), (0,yscale*self.height), linewidth=self.win.dotwidth, color=col, zorder=zorder)
        self.graph = mpl.patches.Polygon([[0,0],[0,0],[0,0]], fill=True, closed=True, zorder=zorder)
        self.win.fig.add_artist(self.rect)
        self.win.fig.add_artist(self.line)
        self.win.fig.add_artist(self.graph)

    def flrange(self, a, b, n):
        delta = (b - a)/n
        s = a
        b += delta/2
        while s < b:
            yield s
            s += delta

    def prepare(self, hist):
        xscale = self.win.xscale()
        yscale = self.win.yscale()
        delta = (hist[0][-1] - hist[0][0])/(2*(len(hist[0])-1))
        x = [hist[0][0]-delta] + list(map(lambda v:v+delta, hist[0]))
        y = list(self.flrange(-1.0 if self.symmetric else 0.0, 1.0, self.num))
        hist2 = [0.0 for i in range(self.num)]
        j = 0
        while j<self.num and y[j+1]<=x[0]:
            j+=1
        left = j
        for i in range(len(hist[0])):
            if y[j+1]>x[i+1]:
                hist2[j] += hist[1][i]
            else:
                hist2[j] += hist[1][i]*(y[j+1]-x[i])/(x[i+1]-x[i])
                j+=1
                while j<self.num and y[j+1]<=x[i+1]:
                    hist2[j] += hist[1][i]*(y[j+1]-y[j])/(x[i+1]-x[i])
                    j+=1
                if j<self.num:
                    hist2[j] += hist[1][i]*(x[i+1]-y[j])/(x[i+1]-x[i])
        right = min(j, self.num-1)
        hmax = max(hist2)
        hist3 = [[(left+1.0)*xscale, 0.0]]
        off = 0
        for i in range(left, right+1):
            if self.symmetric and i*2==self.num:
                off=1
            hist3.append([(i+1.5+off)*xscale, hist2[i]/hmax*self.height*yscale])
        hist3.append([(right+2.0+off)*xscale, 0.0])
        return hist3

    def set(self, hist, tp):
        if hist is not None:
            self.hist = self.prepare(hist)
        self.type = tp
        if not self.hidden:
            if self.type == "Causal":
                self.graph.set_facecolor(hsl_color(causcol, 1.0, -0.2))
            elif self.type == "Direct":
                self.graph.set_facecolor(hsl_color(dircol, 1.0, -0.2))
            elif self.type == "Indirect":
                self.graph.set_facecolor(hsl_color(0.0, 0.0, 0.1))
            else:
                self.graph.set_facecolor(hsl_color(0.0, 0.0, -0.5))

    def show(self):
        if self.hidden:
            self.hidden = False
            self.rect.set_edgecolor('black')
            self.rect.set_facecolor('white')
            if self.symmetric:
                self.line.set_color('black')
            if self.type == "Causal":
                self.graph.set_facecolor(hsl_color(causcol, 1.0, -0.2))
            elif self.type == "Direct":
                self.graph.set_facecolor(hsl_color(dircol, 1.0, -0.2))
            elif self.type == "Indirect":
                self.graph.set_facecolor(hsl_color(0.0, 0.0, 0.1))
            else:
                self.graph.set_facecolor(hsl_color(0.0, 0.0, -0.5))

    def hide(self):
        if not self.hidden:
            self.hidden = True
            col = (0,0,0,0)
            self.rect.set_edgecolor(col)
            self.rect.set_facecolor(col)
            self.line.set_color(col)
            self.graph.set_facecolor(col)

    def move(self, pos):
        self.rect.set_xy(pos)
        self.line.set_xdata((pos[0] + (self.width//2)*self.win.xscale(),)*2)
        self.line.set_ydata((pos[1], pos[1] + self.height*self.win.yscale()))
        self.graph.set_xy(list(map(lambda ele: [ele[0]+pos[0], ele[1]+pos[1]], self.hist)))

    def remove(self):
        self.rect.remove()
        self.line.remove()
        self.graph.remove()


class GraphMessage():
    def __init__(self, win, pos, txt, state = 0, fontsz = 12, statedict = {0:('white','white'), 1:('white', 'black'), 2:(hsl_color(0.0,0.0,0.0),'black'), 3:('black','black')}):
        self.win = win
        self.pos = pos
        self.state = state
        self.statedict = statedict
        bcol = self.statedict[state]
        bpos = win.trans(tupleadd(pos, (fontsz*2/3, fontsz*2/3)))
        tpos = win.trans(tupleadd(pos, (fontsz*2, 0)))
        self.bullet = mpl.patches.Ellipse(bpos, win.xscale()*fontsz, win.yscale()*fontsz, facecolor=bcol[0], edgecolor=bcol[1], linewidth = fontsz/6)
        self.text = plt.Text(tpos[0], tpos[1], txt, fontsize = fontsz)
        win.fig.add_artist(self.bullet)
        win.fig.add_artist(self.text)

    def set_message(self, txt, state):
        if not state == self.state:
            bcol = self.statedict[state]
            self.bullet.set_facecolor(bcol[0])
            self.bullet.set_edgecolor(bcol[1])
            self.state = state
        self.text.set_text(txt)
        #plt.draw()


class GraphWindow():
    def __init__(self, fig, rect):
        self.fig = fig
        self.offx = rect[0]
        self.offy = rect[1]
        inches = fig.get_size_inches()
        self.width = int(fig.dpi*inches[0]*rect[2])
        self.height = int(fig.dpi*inches[1]*rect[3])
        #self.fig.set_size_inches((self.width/self.fig.dpi, self.height/self.fig.dpi))
        #self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)
        #self.fig.canvas.mpl_connect('key_release_event', self.key_release_callback)
        #self.fig.canvas.mpl_connect('scroll_event', self.scroll_callback)
        #self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        #self.fig.canvas.mpl_connect('motion_notify_event', self.button_motion_callback)
        #self.fig.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.nodes = {}
        self.links = {}
        self.hlinks = {}
        self.message = GraphMessage(self, (15, 15), "", state = 0)
        self.dragnode = False
        self.draglink = False
        self.userfunc = False
        self.radius = 24 # 40
        self.thickness = 4
        self.arrowsize = 8
        self.fontsz0 = 10 # 18
        self.fontsz1 = 14 # 24
        self.dotwidth = 0.75
        self.outfile = False
        self.numoutput = 0
        self.continuous = False
        self.convertmi = False
        self.showinterval = False

    def trans(self, pos):
        tr = self.fig.transFigure.inverted().transform(pos)
        return (tr[0]+self.offx, tr[1]+self.offy)

    def xscale(self):
        return self.trans((1,1))[0] - self.trans((0,0))[0]

    def yscale(self):
        return self.trans((1,1))[1] - self.trans((0,0))[1]
        
    def set_output_file(self, filename):
        self.outfile = open(filename, "w")

    def add_node(self, name, x, y):
        GraphNode(self, name, (x, y), self.radius, False) 

    def find_link(self, name1, name2):
        tup = (name1, name2) if name1 < name2 else (name2, name1)
        return self.links[tup] if tup in self.links else False

    def set_link(self, name1, name2, tp, strn):
        link = self.find_link(name1, name2)
        if link is False:
            GraphLink(self, name1, name2, tp, strn)
        else:
            link.modify(name1, name2, tp, strn)

    def find_hlink(self, name1, name2, name3):
        tup = tuple(sorted([name1, name2, name3]))
        return self.hlinks[tup] if tup in self.hlinks else False

    def set_hlink(self, name1, name2, name3, tp, strn):
        link = self.find_hlink(name1, name2, name3)
        if link is False:
            GraphHyperlink(self, name1, name2, name3, tp, strn)
        else:
            link.modify(tp, strn)

    def set_node_prob(self, name1, prob):
        if name1 in self.nodes:
            self.nodes[name1].set_prob(prob)

    def set_node_distr(self, name1, hist, nbins):
        if name1 in self.nodes:
            self.nodes[name1].set_distr(hist, nbins)

    def set_link_distr(self, name1, name2, hist, nbins):
        link = self.find_link(name1, name2)
        if link is not False:
            link.set_distr(hist, nbins)

    def set_hlink_distr(self, name1, name2, name3, hist, nbins):
        link = self.find_hlink(name1, name2, name3)
        if link is not False:
            link.set_distr(hist, nbins)

    def reset(self):
        self.nodes = {}
        self.links = {}
        self.hlinks = {}
        self.fig.clear()
        self.message = GraphMessage(self, (15, 15), "", state = 0)

    def refresh(self):
        #plt.draw()
        pass

    def set_number(self, num, run):
        self.message.set_message(str(num) if num else "", 3 if run else 1 if num else 0)

    def format_strength(self, str):
        if type(str)==tuple:
            if self.showinterval:
                return "[" + format(str[1], ".3f") + ", " + format(str[2], ".3f") + "]"
            else:
                return format(str[0], ".3f")
        else:
            return format(str, ".3f")

    def locate_node(self, event):
        for n in self.nodes:
            if self.nodes[n].circles[0].contains(event)[0]:
                return self.nodes[n]
        return False

    def locate_link(self, event):
        for l in self.links:
            if self.links[l].type not in ['None',''] and self.links[l].isclose(event, self.arrowsize):
                return self.links[l]
        return False

    def locate_hlink(self, event):
        for l in self.hlinks:
            if self.hlinks[l].type not in ['None',''] and self.hlinks[l].path.contains(event)[0]:
                return self.hlinks[l]
        return False

    def allin(self, tlst, lst):
        for e in tlst:
            if not e in lst:
                return False
        return True

    def condprob(self, sam, pa, pb):
        if sam == 0.0:
            pd = pa*pb
        else:
            tt = 2.0*(pa + pb) + 0.5*sam + 0.5/sam - 1.0
            pd = (tt - np.sign(sam)*np.sqrt(tt*tt - 4.0*pa*pb*(sam+1.0)*(sam+1.0)/sam))*0.25
        return [(pd/pa)/(pd/pa + (pb - pd)/(1.0-pa)), (1.0 - pd/pa)/(1.0 - pd/pa + (1.0 - pa - pb + pd)/(1.0 - pa))]

    def save(self, file):
        dd = { 'width': self.width,
               'height': self.height,
               'continuous' : self.continuous,
               'nodes': [ self.nodes[n].todict() for n in self.nodes ],
               'links': [ self.links[l].todict() for l in self.links ] }
        f = open(file, "w")
        f.write(format(dd) + "\n")
        f.close()

    def load(self, file):
        f = open(file, "r")
        s = f.readline()
        dd = eval(s)
        if dd['width'] != self.width or dd['height'] != self.height:
            self.width = dd['width']
            self.height = dd['height']
            self.fig.set_size_inches((self.width/self.fig.dpi, self.height/self.fig.dpi))
        self.continuous = dd['continuous']
        for ele in dd['nodes']:
            self.add_node(ele['name'], ele['pos'][0], ele['pos'][1])
            self.set_node_prob(ele['name'], ele['prob'])
        for ele in dd['links']:
            self.set_link(ele['from'], ele['to'], 'Fixed', ele['strength'])

    def movelimit(self, pos):
        (x,y) = pos
        if x < self.radius:
            x = self.radius
        elif x > self.width-self.radius:
            x = self.width-self.radius
        if y < self.radius:
            y = self.radius
        elif y > self.height-self.radius:
            y = self.height-self.radius
        return (x,y)
        
    def key_press_callback(self, event):
        #print("Key pressed: " + event.key)
        if event.key == "enter" and self.userfunc:
            self.userfunc()

    def key_release_callback(self, event):
        #if event.key == "q":
        pass

    def scroll_callback(self, event):
        pass

    def button_press_callback(self, event):
        if event.button == 1 and event.key == None:
            node = self.locate_node(event)
            if node is not False:
                node.set_selected(True)
                self.dragnode = [node, (node.pos[0] - event.x, node.pos[1] - event.y), (event.x, event.y)]
                plt.draw()
            else:
                link = self.locate_link(event)
                if link is not False:
                    self.nodes[link.fr].set_selected(True)
                    self.nodes[link.to].set_selected(True)
                    self.draglink = [link, (event.x,event.y)]
                    #plt.draw()

    def button_motion_callback(self, event):
        if self.dragnode is not False:
            self.dragnode[0].move(self.movelimit(tupleadd((event.x,event.y), self.dragnode[1])))
            plt.draw()

    def button_release_callback(self, event):
        if event.button == 1:
            if self.dragnode is not False:
                self.dragnode[0].set_selected(False)
                if self.dragnode[2] == (event.x,event.y):
                    for func in self.callbacks['select_attributes']:
                        func([self.dragnode[0].name])
                else:
                    self.dragnode[0].move(self.movelimit(tupleadd((event.x,event.y), self.dragnode[1])))
                plt.draw()
                self.dragnode = False
            elif self.draglink is not False:
                self.nodes[self.draglink[0].fr].set_selected(False)
                self.nodes[self.draglink[0].to].set_selected(False)
                if self.draglink[1] == (event.x,event.y):
                    for func in self.callbacks['select_attributes']:
                        func([self.draglink[0].fr, self.draglink[0].to])
                plt.draw()
                self.draglink = False
            else:
                for func in self.callbacks['select_attributes']:
                    func([])

class GraphVisualizer():
    def __init__(self, fig, rect, repo, callbacks):
        self.callbacks = callbacks
        self.repo = repo
        self.show_hist = False
        # 'select_entity/ies', 'select_attribute/s', 'select_timespan'
        if not 'select_attributes' in self.callbacks:
            self.callbacks['select_attributes'] = []
        self.win = GraphWindow(fig, rect)
        self.win.callbacks = callbacks

    def handle_data(self, vecdic):
        return None

    def redraw_model(self, moddic):
        if moddic['type'] == 'CausalModel':
            nodes = moddic['nodes']
            links = moddic['links']
            hlinks = moddic['hlinks']
            for n in nodes:
                if not n.name in self.win.nodes:
                    self.win.add_node(n.name,
                                      discrete(self.win.width-self.win.radius*2) + self.win.radius,
                                      discrete(self.win.height-self.win.radius*2) + self.win.radius)
                self.win.set_node_prob(n.name, n.prob)
                if self.show_hist and n.hist is not False:
                    self.win.set_node_distr(n.name, n.hist, 60)
                else:
                    pass # remove histogram
            for e in links:
                if e.status == 'Causal':
                    if e.forward:
                        self.win.set_link(e.x1, e.x2, e.status, e.dir)
                    else:
                        self.win.set_link(e.x2, e.x1, e.status, e.dir)
                elif e.status == 'Direct':
                    self.win.set_link(e.x1, e.x2, e.status, e.dir)
                elif e.status == 'Indirect':
                    self.win.set_link(e.x1, e.x2, e.status, e.indir)
                else:
                    self.win.set_link(e.x1, e.x2, 'None', (0.0, 0.0, 0.0, 1.0))
                if self.show_hist and e.hist is not False:
                    self.win.set_link_distr(e.x1, e.x2, e.hist, 60)
                else:
                    pass # remove histogram
            for e in hlinks:
                if e.status == 'Direct':
                    self.win.set_hlink(e.x1, e.x2, e.x3, e.status, e.dir)
                elif e.status == 'Indirect':
                    self.win.set_hlink(e.x1, e.x2, e.x3, e.status, e.indir)
                else:
                    self.win.set_hlink(e.x1, e.x2, e.x3, 'None', (0.0, 0.0, 0.0, 1.0))
                if self.show_hist and e.hist is not False:
                    self.win.set_hlink_distr(e.x1, e.x2, e.x3, e.hist, 60)
                else:
                    pass # remove histogram
        elif moddic['type'] == 'HigherOrderModel':
            relations = moddic['relations']
            for r in relations:
                if not r[0][0] in self.win.nodes:
                    self.win.add_node(r[0][0],
                                      discrete(self.win.width-self.win.radius*2) + self.win.radius,
                                      discrete(self.win.height-self.win.radius*2) + self.win.radius)
                if not r[0][1] in self.win.nodes:
                    self.win.add_node(r[0][1],
                                      discrete(self.win.width-self.win.radius*2) + self.win.radius,
                                      discrete(self.win.height-self.win.radius*2) + self.win.radius)
                self.win.set_link(r[0][0], r[0][1], "Highorder", r[1])


    def redraw_features(self):
        pass

    def default_params(self):
        return {}

    def set_params(self, dic):
        return

    def key_press_event(self, event):
        self.win.key_press_callback(event)

    def key_release_event(self, event):
        self.win.key_release_callback(event)

    def scroll_event(self, event):
        self.win.scroll_callback(event)

    def button_press_event(self, event):
        self.win.button_press_callback(event)

    def motion_notify_event(self, event):
        self.win.button_motion_callback(event)

    def button_release_event(self, event):
        self.win.button_release_callback(event)
