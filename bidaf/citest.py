# Todo:
# Various continuous tests:
#   Split at m places and use n-ary attributes, requires solving integrals
# Test various integrals on binary data, test highorder accuracy
# 191029:
#  - Higher order doesnt work, check IPPF and integration scheme
#  - Tolerance for correlation
#  - Extrapolation for discretization of continuous values

from math import * 
from random import * 
import numpy as np
import scipy.stats
from scipy.special import erfinv


global debugdata

def get_debug():
    global debugdata
    return debugdata

#-------------------
# utility functions

def depth(a):
    return 1 + depth(a[0]) if type(a) is list else 0

def dims(a):
    return dims(a[0]) + [len(a)] if type(a) is list else []

def dimsprod(a):
    return dimsprod(a[0]) * len(a) if type(a) is list else 1

def deepcopy(lst):
    return list(map(deepcopy, lst)) if type(lst)==list else lst

def square(x):
    return x*x

def dotp(v1,v2):
    sum = 0.0
    for i in range(min(len(v1),len(v2))):
        sum += v1[i]*v2[i]
    return sum

def safelog(x):
    return log(x) if x>0 else -inf

def safesqrt(x):
    return sqrt(x) if x > 0 else 0.0

def listpart(lst, inds):
    if type(inds)==int:
        inds = (inds,)
    return [lst[i] for i in inds] 

def find_solution(func, a, b, val, eps):
    fa = func(a)
    fb = False
    m = 0.5*(a+b)
    fm = func(m)
    while abs(fm-val) > eps:
        if ((fm > val) == (fa > fm)):
            fa = fm
            a = m
        else:
            fb = fm
            b = m
        m = 0.5*(a+b)
        fm = func(m)
    return m

def find_peak(func, a, b, mindiff):
    fa = np.nan
    fb = np.nan
    m1 = (a+a+b)/3.0
    fm1 = func(m1)
    m2 = (a+b+b)/3.0
    fm2 = func(m2)
    left = []
    right = []
    while True:
        if fm1 > fm2:
            right = [(b, fb)] + right
            (m, fm) = (m1, fm1)
            (b, fb) = (m2, fm2)
        else:
            left = [(a, fa)] + left
            (a, fa) = (m1, fm1)
            (m, fm) = (m2, fm2)
        if np.isnan(fa):
            tmp = [fm, fb, right[0][1]]
            dir = 0
        elif np.isnan(fb):
            tmp = [fm, fa, left[0][1]]
            dir = 1
        else:
            tmp = [fa, fm, fb]
            dir = 0 if fa > fb else 1
#        print(a, fa, m, fm, b, fb)
        mx = max(tmp)
        mn = min(tmp)
        if mx-mn < mindiff:
            break
        if (m-a) > (b-m)*1.5:
            dir = 0
        elif (m-a)*1.5 < (b-m):
            dir = 1
        if dir == 0:
            m1 = 0.5*(a+m)
            fm1 = func(m1)
            m2 = m
            fm2 = fm
        else:
            m1 = m
            fm1 = fm
            m2 = 0.5*(m+b)
            fm2 = func(m2)
    if mx == fa:
        return (a, fa)
    elif mx == fb:
        return (b, fb)
    else:
        return (m, fm)

def find_irange(func, a, b, mindiff, maxdiff):
    fa = np.nan
    fb = np.nan
    m1 = (a+a+b)/3.0
    fm1 = func(m1)
    m2 = (a+b+b)/3.0
    fm2 = func(m2)
    left = []
    right = []
    while True:
        if fm1 > fm2:
            right = [(b, fb)] + right
            (m, fm) = (m1, fm1)
            (b, fb) = (m2, fm2)
        else:
            left = [(a, fa)] + left
            (a, fa) = (m1, fm1)
            (m, fm) = (m2, fm2)
        if np.isnan(fa):
            tmp = [fm, fb, right[0][1]]
            dir = 0
        elif np.isnan(fb):
            tmp = [fm, fa, left[0][1]]
            dir = 1
        else:
            tmp = [fa, fm, fb]
            dir = 0 if fa > fb else 1
#        print(a, fa, m, fm, b, fb)
        mx = max(tmp)
        mn = min(tmp)
        if mx-mn < mindiff:
            break
        if (m-a) > (b-m)*1.5:
            dir = 0
        elif (m-a)*1.5 < (b-m):
            dir = 1
        if dir == 0:
            m1 = 0.5*(a+m)
            fm1 = func(m1)
            m2 = m
            fm2 = fm
        else:
            m1 = m
            fm1 = fm
            m2 = 0.5*(m+b)
            fm2 = func(m2)
    delta = min(m-a, b-m)
    tr = fm - maxdiff
    if len(left):
        leftind = 0
        while not np.isnan(left[leftind][1]) and left[leftind][1] > tr:
            leftind += 1
        a = left[leftind][0]
        fa = left[leftind][1]
    if np.isnan(fa):
        a += delta
        fa = func(a)
    if len(right):
        rightind = 0
        while not np.isnan(right[rightind][1]) and right[rightind][1] > tr:
            rightind += 1
        b = right[rightind][0]
        fb = right[rightind][1]
    if np.isnan(fb):
        b -= delta
        fb = func(b)
    return (a, b, m, delta)

def find_calc_irange(func, a, b, mindiff, maxdiff):
    fa = np.nan
    fb = np.nan
    m1 = (a+a+b)/3.0
    fm1 = func(m1)
    m2 = (a+b+b)/3.0
    fm2 = func(m2)
    left = []
    right = []
    while True:
        if fm1 > fm2:
            right = [(b, fb)] + right
            (m, fm) = (m1, fm1)
            (b, fb) = (m2, fm2)
        else:
            left = [(a, fa)] + left
            (a, fa) = (m1, fm1)
            (m, fm) = (m2, fm2)
        if np.isnan(fa):
            tmp = [fm, fb, right[0][1]]
            dir = 0
        elif np.isnan(fb):
            tmp = [fm, fa, left[0][1]]
            dir = 1
        else:
            tmp = [fa, fm, fb]
            dir = 0 if fa > fb else 1
        mx = max(tmp)
        mn = min(tmp)
        if mx-mn < mindiff:
            break
        if (m-a) > (b-m)*1.5:
            dir = 0
        elif (m-a)*1.5 < (b-m):
            dir = 1
        if dir == 0:
            m1 = 0.5*(a+m)
            fm1 = func(m1)
            m2 = m
            fm2 = fm
        else:
            m1 = m
            fm1 = fm
            m2 = 0.5*(m+b)
            fm2 = func(m2)
    delta = min(m-a, b-m)
    left = [(a, fa)] + left
    right = [(b, fb)] + right
    if m-a < m-b:
        right = [(m, fm)] + right
    else:
        left = [(m, fm)] + left
    tr = mx - maxdiff
    resx = []
    resy = []
    leftind = 0
    while not np.isnan(left[leftind][1]) and left[leftind][1] > tr:
        resx = [left[leftind][0]] + resx
        resy = [left[leftind][1]] + resy
        a = left[leftind][0] - delta
        leftind += 1
        while a > left[leftind][0] + delta/2:
            fa = func(a)
            if np.isnan(fa) or fa < tr:
                break
            resx = [a] + resx
            resy = [fa] + resy
            a -= delta
    rightind = 0
    while not np.isnan(right[rightind][1]) and right[rightind][1] > tr:
        resx = resx + [right[rightind][0]]
        resy = resy + [right[rightind][1]]
        b = right[rightind][0] + delta
        rightind += 1
        while b < right[rightind][0] - delta/2:
            fb = func(b)
            if np.isnan(fb) or fb < tr:
                break
            resx = resx + [b]
            resy = resy + [fb]
            b += delta
    return (resx, resy)


#-------------------
# Data preparing functions

def get_mean_data(data, attrs):
    num = len(data)
    mean = [0.0 for ind in attrs]
    for vec in data:
        for i in range(len(attrs)):
            mean[i] += vec[attrs[i]]
    if num > 0.0:
        for i in range(len(attrs)):
            mean[i] /= num
    return mean

def get_covar_data(data, attrs):
    num = len(data)
    mean = get_mean_data(data, attrs)
    covar = [[0.0 for i in attrs] for j in attrs]
    for vec in data:
        for i in range(len(attrs)):
            for j in range(len(attrs)):
                covar[i][j] += (vec[attrs[i]] - mean[i])*(vec[attrs[j]] - mean[j])
    if num > 0.0:
        for i in range(len(attrs)):
            for j in range(len(attrs)):
                covar[i][j] /= num
    return covar

def get_trivar_data(data, attrs):
    num = len(data)
    mean = get_mean_data(data, attrs)
    trivar = [[[0.0 for i in attrs] for j in attrs] for k in attrs]
    for vec in data:
        for i in range(len(attrs)):
            for j in range(len(attrs)):
                for k in range(len(attrs)):
                    trivar[i][j][k] += (vec[attrs[i]] - mean[i])*(vec[attrs[j]] - mean[j])*(vec[attrs[k]] - mean[k])
    if num > 0.0:
        for i in range(len(attrs)):
            for j in range(len(attrs)):
                for k in range(len(attrs)):
                    trivar[i][j][k] /= num
    return trivar

def get_trivar_cond_data(data, attrs, cond):
    num = len(data)
    cov = get_covar_data(data, attrs + cond)
    mean = get_mean_data(data, attrs)
    if len(cond) > 0:
        inv = matrix_inverse(cov)
        c = [[inv[i][j] for i in range(len(attrs))] for j in range(len(attrs))]
        b = [[inv[i][j] for i in range(len(attrs), len(attrs)+len(cond))] for j in range(len(attrs))]
        cov = matrix_inverse(c)
        mean0 = mean
        m = np.matmul(cov, b)
    std = [sqrt(cov[i][i]) for i in range(len(attrs))]
    trivar = 0.0
    for vec in data:
        if len(cond) > 0:
            mean = mean0 - np.matmul(m, [vec[i] for i in cond])
        trivar += (vec[attrs[0]] - mean[0])/std[0]*(vec[attrs[1]] - mean[1])/std[1]*(vec[attrs[2]] - mean[2])/std[2]
    if num > 0.0:
        trivar /= num
    return trivar

def incrlistpart(lst, args):
    for ind in args[:-1]:
        lst = lst[ind]
    lst[args[-1]] += 1

def incrlistpart_weightedpairs(lst, args, weight):
    # args is list of (ind, w1, w2) or just ind
    if len(args)==1:
        if type(args[0])==int:
            lst[args[0]] += weight
        else:
            lst[args[0][0]] += weight*args[0][1]
            lst[args[0][0]+1] += weight*args[0][2]
    else:
        if type(args[0])==int:
            incrlistpart_weightedpairs(lst[args[0]], args[1:], weight)
        else:
            incrlistpart_weightedpairs(lst[args[0][0]], args[1:], weight*args[0][1])
            incrlistpart_weightedpairs(lst[args[0][0]+1], args[1:], weight*args[0][2])

def get_hc_data(data, attrs):
    cube = 0
    for ind in attrs:
        cube=[cube, deepcopy(cube)]
    for vec in data:
        incrlistpart(cube, list(map(lambda ind:vec[ind], attrs)))
    return cube

def get_nhc_data(data, attrs, dim):
#    cube = 0
#    d = len(attrs)
#    for i in range(d):
#        newcube=[cube]
#        for j in range(1,dim[d-i-1]):
#            newcube += [deepcopy(cube)]
#   cube = newcube
    cube = make_hc(dim)
    for vec in data:
        incrlistpart(cube, list(map(lambda ind:vec[ind], attrs)))
    return cube

def get_general_hc_data(data, form, attrs, cond, pseudo, lev):
    # Anta form[i] = [('bin',2),('discr',n),('symbol',[lst]),('cont', [ecd])]
    global discretization
    dim = [False]*(len(attrs)+len(cond))
    for i in range(len(cond)):
        dim[i] = discretize_dim(form[cond[i]], discretization<<lev)
    for i in range(len(attrs)):
        dim[i+len(cond)] = discretize_dim(form[attrs[i]], discretization)
    cube = make_hc(dim)
    for vec in data:
        if pseudo:
            incrlistpart_weightedpairs(cube,
                                       [discretize_value_pseudo(vec[ind], form[ind], discretization<<lev) for ind in cond] +
                                       [discretize_value_pseudo(vec[ind], form[ind], discretization) for ind in attrs],
                                       1.0)
        else:
            incrlistpart(cube,
                         [discretize_value(vec[ind], form[ind], discretization<<lev) for ind in cond] +
                         [discretize_value(vec[ind], form[ind], discretization) for ind in attrs])
    return cube

def get_general_hc_data_alt(data, form, attrs, discr, pseudo): # attrs = cond + attrs
    # Anta form[i] = [('bin',2),('discr',n),('symbol',[lst]),('cont', [ecd])]
    global discretization
    dim = [False]*(len(attrs))
    for i in range(len(attrs)):
        dim[i] = discretize_dim(form[attrs[i]], discr[i])
    cube = make_hc(dim)
    for vec in data:
        if pseudo:
            incrlistpart_weightedpairs(cube,
                                       [discretize_value_pseudo(vec[ind], form[ind], discr[i]) for i,ind in enumerate(attrs)],
                                       1.0)
        else:
            incrlistpart(cube,
                         [discretize_value(vec[ind], form[ind], discr[i]) for i,ind in enumerate(attrs)])
    return cube

def make_ecd(data, attr, num):
    values = [vec[attr] for vec in data]
    values.sort()
    n = len(values)
    dd = [0]*(num-1)
    for i in range(num-1):
        dd[i] = values[n*(i+1)//num]
    return dd

def twoflat(lst):
    if depth(lst)==2:
        return [lst[0]+lst[1]]
    else:
        return twoflat(lst[0]) + twoflat(lst[1])

def kflat(lst, k):
    if depth(lst)==k:
        return [lst]
    else:
        return kflat(lst[0], k) + kflat(lst[1], k)

def nkflat(lst, k):
    if depth(lst)==k:
        return [lst]
    else:
        ret = []
        for l in lst:
            ret += nkflat(l, k)
        return ret

# preformat: ['bin', 'discr', 'symbol', 'cont', ...]
# old format: [('nary', n), ('discr', n, {'val1':0, 'val2':1, ...}), ('cont', n, {(v1, v2):0, (v2, v3):1, ...}), ...]
# new format: [('bin', 2), ('nary', n), ('discr', [list]), ('symbol', [list]), ('cont', [ecd])]

discretization = 5

def discretization_():
    global discretization
    return discretization

def set_discretization_(d):
    global discretization
    discretization = d

def make_format(data, preformat):
    form = []
    for i in range(len(preformat)):
        if preformat[i] == 'bin':
            form.append(('bin', 2))
        elif preformat[i] == 'nary':
            mx = 0
            for vec in data:
                if vec[i] > mx:
                    mx = vec[i]
            form.append(('nary', mx+1))
        elif preformat[i] == 'discr' or preformat[i] == 'symbol':
            values = []
            for vec in data:
                if vec[i] not in values:
                    values.append(vec[i])
            values.sort()
            form.append((preformat[i], values))
        elif preformat[i]== 'cont':
            form.append(('cont', make_ecd(data, i, 120)))
#        elif preformat[i] == 'discr':
#            values = []
#            for vec in data:
#                if vec[i] not in values:
#                    values.append(vec[i])
#            values.sort()
#            form.append(('discr', len(values), { values[j]:j for j in range(len(values)) }))
#        elif preformat[i]== 'cont':
#            form.append(('cont', discretization, make_cont_format_1(data, i, discretization)))
        else:
            form.append(False)
    return form

def make_cont_format_1(data, attr, num):
    dd = {}
    values = [vec[attr] for vec in data]
    values.sort()
    n = len(values)
    v1 = 0
    v2 = -inf
    for i in range(num):
        v1 = v2
        v2 = values[n*(i+1)//num] if i+1 < num else inf
        dd[(v1,v2)] = i
    return dd

def make_cont_format_2(data, attr, num):
    dd = {}
    values = [vec[attr] for vec in data]
    mean = sum(values)/len(values)
    std = sqrt(sum([square(val-mean) for val in values]))
    v1 = 0
    v2 = -inf
    for i in range(num):
        v1 = v2
        v2 = mean + std*erfinv(2*(i+1.0)/num-1)
        dd[(v1,v2)] = i
    return dd

def make_cont_format_3(data, attr, num):
    pass

def discretize_dim(formspec, discr):
    if formspec[0] == 'nary' or formspec[0] == 'bin':
        return formspec[1]
    elif formspec[0] == 'discr' or formspec[0] == 'symbol':
        return len(formspec[1])
    elif formspec[0] == 'cont':
        return discr
    else:
        return 1

def discretize_value(val, formspec, discr):
    if formspec[0] == 'nary' or formspec[0] == 'bin':
        return val
    elif formspec[0] == 'discr' or formspec[0] == 'symbol':
        return formspec[1].index(val)
    elif formspec[0] == 'cont':
        n = len(formspec[1])+1
        for i in range(1,discr):
            if val < formspec[1][n*i//discr]:
                return i-1
        return discr-1
    else:
        return 0

def discretize_value_pseudo(val, formspec, discr):
    if formspec[0] == 'nary' or formspec[0] == 'bin':
        return val
    elif formspec[0] == 'discr' or formspec[0] == 'symbol':
        return formspec[1].index(val)
    elif formspec[0] == 'cont':
        # returns tuple (ind, w1, w2)
        n = len(formspec[1])+1
        ind = discr-2
        w = 1.0
        for i in range(discr-1):
            if val < formspec[1][n*(2*i+3)//(2*discr)]:
                ind = i
                fr = n*(2*i+1)//(2*discr)
                to = n*(2*i+3)//(2*discr)
                if i==0 and val < formspec[1][fr]:
                    w = 0.0
                    break
                for j in range(fr, to):
                    if val < formspec[1][j+1]:
                        y = (val - formspec[1][j])/(formspec[1][j+1] - formspec[1][j]) + j
                        w = (y - fr) / (to - fr)
                        break
                break
        return (ind, 1.0-w, w)
    else:
        return 0

# old discretization scheme, but better at accumulation points
def discretize_sample(vec, form):
    res = []
    for i in range(len(form)):
        if form[i][0] == 'nary' or form[i][0] == 'bin':
            res.append(vec[i])
        elif form[i][0] == 'discr':
            res.append(form[i][2][vec[i]])
        elif form[i][0] == 'cont':
            for k in form[i][2]:
                if vec[i] <= k[1]:
                    if vec[i] >= k[0] and (vec[i] < k[1] or k[0] == k[1]):
                        res.append(form[i][2][k])
                        break
    return res

# old discretization scheme
def discretize_data(data, form):
    return ([('nary', f[1]) for f in form],
            [discretize_sample(vec, form) for vec in data])

#-------------------
# Histogram functions

def hist_peak(hist):
    mx = 0.0
    mxind = False
    for i in range(len(hist[0])):
        if mx < hist[1][i]:
            mx = hist[1][i]
            mxind = hist[0][i]
    return mxind

def hist_significance(hist, val, peak, tol = 0.0):
    sum = 0.0
    if val > peak:
        for i in range(len(hist[0])):
            if hist[0][i] >= val - tol:
                sum += hist[1][i]
    else:
        for i in range(len(hist[0])):
            if hist[0][i] <= val + tol:
                sum += hist[1][i]
    return sum

def hist_mean_var(hist):
    sum = 0
    sqsum = 0
    for i in range(len(hist[0])):
        sum += hist[0][i]*hist[1][i]
        sqsum += hist[0][i]*hist[0][i]*hist[1][i]
    return (sum, sqsum-sum*sum)

def hist_maxinterval(hist, mass):
    i = 0
    j = len(hist[0])-1
    while mass < 1 and i < j:
        if hist[1][i] < hist[1][j]:
            mass += hist[1][i]
            i += (1 if mass < 1 else 0)
        else:
            mass += hist[1][j]
            j -= (1 if mass < 1 else 0)
    return (hist[0][i], hist[0][j])


#-------------------
# G2 test

def mutualinfo(nlst):
    ntot = sum(nlst)
    n1 = nlst[0]+nlst[1]
    n2 = nlst[0]+nlst[2]
    return (nlst[0]*safelog(nlst[0]*ntot/(n1*n2)) if nlst[0] else 0.0) + \
        (nlst[1]*safelog(nlst[1]*ntot/(n1*(ntot-n2))) if nlst[1] else 0.0) + \
        (nlst[2]*safelog(nlst[2]*ntot/((ntot-n1)*n2)) if nlst[2] else 0.0) + \
        (nlst[3]*safelog(nlst[3]*ntot/((ntot-n1)*(ntot-n2))) if nlst[3] else 0.0)

def g2_signif(data, attrs, cond):
    d = get_hc_data(data, cond + attrs)
    v = twoflat(d)
    cmi = 2*sum(list(map(mutualinfo, v)))
    return (1.0 - scipy.stats.chi2.cdf(cmi, len(v)), cmi/(2*sum(list(map(sum, v)))))


#-------------------
# Fisher exact test

def fisher_signif(data, attrs, cond):
    d = get_hc_data(data, cond + attrs)
    v = twoflat(d)
    lst = list(map(lambda vv : scipy.stats.fisher_exact([vv[0:2],vv[2:4]]), v))
    return (min(list(map(lambda x:x[1], lst))),
            sum(list(map(lambda x:(sqrt(x[0])-1.0)/(sqrt(x[0])+1.0), lst)))/len(lst))

# non-used version

def myhyper(nlst):
    if nlst[0]*nlst[3] > nlst[1]*nlst[2]:
        a1 = 1-nlst[0]
        a2 = 1-nlst[3]
        b1 = nlst[1]+2
        b2 = nlst[2]+2
    else:
        a1 = 1-nlst[1]
        a2 = 1-nlst[2]
        b1 = nlst[0]+2
        b2 = nlst[3]+2
    res = 1.0
    for i in range(min(-a1, -a2))[::-1]:
        res = 1.0 + res * (a1+i)*(a2+i)/(b1+i)/(b2+i)
    return res

def classical_sign1(nlst):
    lh = scipy.special.gammaln(nlst[0]+nlst[1]+1) + scipy.special.gammaln(nlst[0]+nlst[2]+1) + \
         scipy.special.gammaln(nlst[2]+nlst[3]+1) + scipy.special.gammaln(nlst[1]+nlst[3]+1) + \
         scipy.log(myhyper(nlst)) - \
         scipy.special.gammaln(nlst[0]) - scipy.special.gammaln(nlst[3]) - scipy.special.gammaln(nlst[1]+2) - \
         scipy.special.gammaln(nlst[2]+2) - scipy.special.gammaln(nlst[0]+nlst[1]+nlst[2]+nlst[3]+1)
    return 1.0 - scipy.exp(lh) if lh < 0 else 0.0 

def classical_sign(nlsts):
    return min(list(map(lambda lst : classical_sign1(lst), nlsts)))


#-------------------
# Sam related tests

# Hypercube functions

def make_hc(dim, ele=0):
    cube = ele
    d = len(dim)
    for i in range(d):
        newcube = [cube]
        for j in range(1, dim[d-i-1]):
            newcube += [deepcopy(cube)]
        cube = newcube
    return cube

def hc_part(cube, ind):
    if type(ind) == int:
        return cube[ind]
    elif len(ind)==1:
        return cube[ind[0]]
    else:
        return hc_part(cube[ind[-1]], ind[:-1])

def hc_setpart(cube, ind, val):
    if type(ind) == int:
        cube[ind] = val
    elif len(ind)==1:
        cube[ind[0]] = val
    else:
        hc_setpart(cube[ind[-1]], ind[:-1], val)

def hc_range(nums):
    dim = len(nums)
    inds = [0]*dim
    ok = True
    while ok:
        yield inds
        k = 0
        while k < dim and inds[k]+1 >= nums[k]:
            inds[k] = 0
            k += 1
        if k < dim:
            inds[k] += 1
        else:
            ok = False

def hc_range_no2(nums):
    dim = len(nums)
    inds = [0]*dim
    nums = [n if n > 2 else 1 for n in nums]
    ok = True
    while ok:
        yield inds
        k = 0
        while k < dim and inds[k]+1 >= nums[k]:
            inds[k] = 0
            k += 1
        if k < dim:
            inds[k] += 1
        else:
            ok = False

def hc_indmap(func, cube, ind=[]):
    if type(cube) == list:
        return [hc_indmap(func, cube[i], [i]+ind) for i in range(len(cube))]
    else:
        return func(cube, ind)

def hc_levelsum(cube):
    if type(cube[0]) == list:
        return [hc_levelsum(list(map(lambda lst:lst[i], cube))) for i in range(len(cube[0]))]
    else:
        return sum(cube)

def hc_equal(cube1, cube2, rel):
    if type(cube1) == list:
        for i in range(len(cube1)):
            if not hc_equal(cube1[i], cube2[i], rel):
                return False
        return True
    else:
        return abs(cube1 - cube2) <= rel*cube1

def hc_lintrans(cube, a, b):
    if type(cube)==list:
        return list(map(lambda lst: hc_lintrans(lst, a, b), cube))
    else:
        return cube*a + b

def hc_subsets(lst, ord):
    if ord==0:
        return [[]]
    elif  ord > len(lst):
        return []
    else:
        rest = hc_subsets(lst[1:], ord-1)
        return list(map(lambda l:[lst[0]]+l, rest)) + hc_subsets(lst[1:],ord)

def hc_dotp(cube1, cube2):
    if type(cube1)==list:
        return sum(map(hc_dotp, cube1, cube2))
    else:
        return cube1*cube2
    

def hc_outer(margins):
    if len(margins)==1:
        return margins[0]
    else:
        cube = hc_outer(margins[:-1])
        return [hc_lintrans(cube, w, 0) for w in margins[-1]]

def hc_margin(cube, marg):
    if type(cube)==list:
        if depth(cube)-1 in marg:
            return list(map(lambda lst: hc_margin(lst, marg), cube))
        else:
            return hc_margin(hc_levelsum(cube), marg)
    else:
        return cube

def hc_scale(cube, marg, sc):
    if type(cube)==list:
        if depth(cube)-1 in marg:
            return list(map(lambda lst,s: hc_scale(lst, marg, s), cube, sc))
        else:
            return list(map(lambda lst: hc_scale(lst, marg, sc), cube))
    else:
        return cube*sc

def hc_scalei(cube, marg, sc, fact):
    if type(cube)==list:
        if depth(cube)-1 in marg:
            return list(map(lambda lst,s: hc_scalei(lst, marg, s, fact), cube, sc))
        else:
            return list(map(lambda lst: hc_scalei(lst, marg, sc, fact), cube))
    else:
        return cube*fact/sc if sc != 0 else 0

def hc_scale2(cube, marg, sc, isc):
    if type(cube)==list:
        if depth(cube)-1 in marg:
            return list(map(lambda lst,s,i: hc_scale2(lst, marg, s, i), cube, sc, isc))
        else:
            return list(map(lambda lst: hc_scale2(lst, marg, sc, isc), cube))
    else:
        return cube*sc/isc if not isc==0.0 else 0.0

def hc_wsum(cube1, cube2, w1, w2):
    if type(cube1)==list:
        return list(map(lambda c1,c2: hc_wsum(c1,c2,w1,w2), cube1, cube2))
    else:
        return cube1*w1 + cube2*w2

#def hc_avg(cube1, cube2):
#    if type(cube1)==list:
#        return list(map(lambda c1,c2: hc_avg(c1,c2), cube1, cube2))
#    else:
#        return (cube1 + cube2)/2

def hc_sqdist(cube1, cube2):
    if type(cube1)==list:
        return sum(map(hc_sqdist, cube1, cube2))
    else:
        return square(cube1-cube2)

def hc_ippf(cube, names, sdict):
    ndict = {}
    i=0
    for n in names:
        ndict[n] = i
        i += 1
    minds = hc_subsets(names,len(names)-1)
    if len(names) == 2:
        margs = list(map(lambda lst:sdict[lst[0]], minds))
    else:
        margs = list(map(lambda lst:sdict[tuple(lst)], minds))
    minds = [[ndict[n] for n in m] for m in minds]
    iter = 0
    while iter < 1000:
        last = cube
        for i in range(len(margs)):
            cube = hc_scale2(cube, minds[i], margs[i], hc_margin(cube, minds[i]))
        if hc_equal(cube, last, 1e-5):
#            print(iter)
            return cube
        iter += 1
    print("IPPF failed to converge")
    return hc_wsum(cube, last, 0.5, 0.5)

def hc_uniippf(cube):
    num = depth(cube)
    dm = dims(cube)
    dm.reverse()
    minds = hc_subsets(list(range(num)), num-1)
    iter = 0
    while iter < 1000:
        last = cube
        for i in range(num):
            cube = hc_scalei(cube, minds[i], hc_margin(cube, minds[i]), dm[i])
        if hc_equal(cube, last, 1e-5):
#            print(iter)
            return cube
        iter += 1
    print("IPPF failed to converge")
    return hc_wsum(cube, last, 0.5, 0.5)

def hc_internrescale(val, ind, ov, nv, ind0, dim):
    mass = 1
    sign = 1
    for i in range(len(ind0)):
        if ind[i] != ind0[i]:
            mass *= dim[i]
            sign *= -1
    return (1+val)*(mass+sign*nv)/(mass+sign*ov)-1

# elementet inds ska sättas till low, allt annat balanseras om
# (0.2, -0.3, 0.1) -> (-1, 0.17, 0.83) 0.7,1.1*(ny summa:2.0--1.0)/(gammal summa:2.0-0.2)
def hc_simplexlower(cube, inds):
    oldval = hc_part(cube, inds)
    dim = dims(cube)
    return hc_indmap(lambda val,ind: hc_internrescale(val, ind, oldval, -1.0, inds, dim), cube)

def hc_simplexheighten(cube, inds):
    oldval = hc_part(cube, inds)
    dim = dims(cube)
    high = min(dim)-1.0
    return hc_indmap(lambda val,ind: hc_internrescale(val, ind, oldval, high, inds, dim), cube)

def hc_binarize(cube, vals):
#    global debugdata
#    debugdata = (cube, vals)
    if type(cube)==list:
        rvals = vals[:-1]
        cube1 = hc_binarize(cube[vals[-1]], rvals)
        cuber = hc_lintrans(cube1, 0, 0)
        for i in range(len(cube)):
            if i != vals[-1]:
                cuber = hc_wsum(cuber, hc_binarize(cube[i], rvals), 1, 1)
        return [cube1, cuber]
    else:
        return cube

# sam2 functions

def sam2_prob(s, px, py):
    if s==0.0:
        pd = px*py
    else:
        tmp = 0.5*(1.0/s+s) + 2*(px+py) - 1.0
        pd = (tmp + (-1 if s > 0.0 else 1)*sqrt(-4*px*py*(1+s)*(1+s)/s + tmp*tmp))/4
    return [pd, px-pd, py-pd, 1-px-py+pd]

def sam2_from_or(q):
    return (sqrt(q)-1)/(sqrt(q)+1)

def integrate2_range(nlst, llim, nstep):
    ntot = sum(nlst)
    if ntot==0: ntot=1
    n1 = nlst[0]+nlst[1]
    n2 = nlst[0]+nlst[2]
    lnp1 = lambda p: n1*safelog(p) + (ntot-n1)*safelog(1-p)
    lnp2 = lambda p: n2*safelog(p) + (ntot-n2)*safelog(1-p)
    e1 = lnp1(n1/ntot)
    e2 = lnp2(n2/ntot)
    plow1 = find_solution(lnp1, n1/ntot, 0.0, e1-llim, 0.1)
    phigh1 = find_solution(lnp1, n1/ntot, 1.0, e1-llim, 0.1)
    plow2 = find_solution(lnp2, n2/ntot, 0.0, e2-llim, 0.1)
    phigh2 = find_solution(lnp2, n2/ntot, 1.0, e2-llim, 0.1)
    pdiff1 = (phigh1-plow1)/nstep
    pdiff2 = (phigh2-plow2)/nstep
    delta = safelog(pdiff1*pdiff2)
    py = plow2+pdiff2/2
    while py < phigh2:
        px = plow1+pdiff1/2
        while px < phigh1:
            yield (px, py, delta)
            px += pdiff1
        py += pdiff2

def sam2_srange(sfunc):
#    a, b, delta = find_irange(sfunc, -1, 1, 0.1, 20)
    a, b, m, delta = find_irange(sfunc, -1, 1, 0.2, 12)
    s = a
    while s <= b:
        yield s
        s += delta
        if abs(s) < delta*1e-4:
            s = 0.0

def integrate2_pxpy_log(s, nlst, rng):
    res=[dotp(nlst, list(map(safelog,sam2_prob(s,px,py)))) + delta
         for px,py,delta in rng]
    mx = max(res)
    sm = 0.0
    for i in range(len(res)):
        sm += exp(res[i] - mx)
    return safelog(sm) + mx

def normalize_logprob(vec):
    mx = max(vec)
    sm = 0.0
    for i in range(len(vec)):
        sm += exp(vec[i] - mx)
    norm = safelog(sm) + mx
    return [exp(x-norm) for x in vec]

def sam2_histogram(nlsts):
    rngs = list(map(lambda nlst: list(integrate2_range(nlst,14,16)), nlsts))
    sfunc = lambda s: sum(map(lambda nlst,rng: integrate2_pxpy_log(s, nlst, rng), nlsts, rngs))
    (srng,res) = find_calc_irange(sfunc, -1, 1, 0.2, 12)
    return [srng, normalize_logprob(res)]
#    srng = list(sam2_srange(sfunc))
#    res = [sfunc(s) for s in srng]
#    mx = max(res)
#    sm = 0.0
#    for i in range(len(res)):
#        sm += exp(res[i] - mx)
#    norm = safelog(sm) + mx
#    return [srng, list(map(lambda x:exp(x-norm), res))]

# Sam_nk functions

def samnk_from_prob(names, probs):
    # From the k-order hypercube probs of probabilities, with dim n1, n2, ...nk on the axes,
    # create a dictionary of (redundant) hypercubes representing all interactions up to order k
    # This is done by running IPPF with completely uniform k-1 order marginals.
    dict={}
    idx=list(range(len(names)))
    # hitta alla subset av [0, (k-1)] dvs marginaler
    # ta fram marginalen, kör uniippf på den, normalisera, stoppa in i dict
    for j in idx:
        marg = hc_margin(probs, [j])
        dict[names[j]] = marg
    for i in range(2,len(names)):
        for set in hc_subsets(idx, i):
            marg = hc_margin(probs, set)
            dict[tuple(map(lambda j: names[j], set))] = hc_lintrans(hc_uniippf(marg), 1.0, -1.0)
    dict[tuple(names)] = hc_lintrans(hc_uniippf(probs), 1.0, -1.0)
    return dict

def samnk_prob(sdict):
    # Take interaction parameters into probability distribution
    # Find max order (highest ind in dict). For each order, fix all marginal probs. Done.
    names = []
    for key in sdict:
        if type(key)==tuple and len(key) > len(names):
            names = list(key)
    idx = list(range(len(names)))
    pdict = {}
    for j in idx:
        pdict[names[j]] = sdict[names[j]]
    for i in range(2,len(names)):
        for set in hc_subsets(idx, i):
            tup = tuple(map(lambda j: names[j], set))
            marg = sdict[tup]
            sz = dimsprod(marg)
            marg = hc_lintrans(marg, 1/sz, 1/sz) 
            pdict[tup] = hc_ippf(marg, list(tup), pdict)
    tup = tuple(names)
    marg = sdict[tup]
    sz = dimsprod(marg)
    marg = hc_lintrans(marg, 1/sz, 1/sz)
    return list(tup), hc_ippf(marg, list(tup), pdict)

def samnk_loglikelihood(probs, counts):
    if type(counts)==list:
        return sum(map(lambda p,c : samnk_loglikelihood(p, c), probs, counts))
    else:
        return safelog(probs)*counts

def samnk_stoll(slst, mlst, counts):
    sdict = { mlst[i] : slst[i] for i in range(len(slst))}
    names, probs = samnk_prob(sdict)
    return samnk_loglikelihood(probs, counts)

def integrate_n1_range(nlst, llim):
    pass
#    ntot = sum(nlst)
#    if ntot==0: ntot=1
#    n = nlst[0]
#    lnp = lambda p: n*safelog(p) + (ntot-n)*safelog(1-p)
#    e = lnp(n/ntot)
#    plow = find_solution(lnp, n/ntot, 0.0, e-llim, 0.1)
#    phigh = find_solution(lnp, n/ntot, 1.0, e-llim, 0.1)
#    return [plow, phigh]

def integrate_n1_peak(nlst):
    nsum = sum(nlst) + len(nlst)
    return list(map(lambda n:(n+1)/nsum, nlst))

def integrate_n1b_peak(nlst, dim):
    return (nlst[0]+1)/(sum(nlst)+dim)

def s2tohc(s, k):
    if k==1:
        return [s, -s]
    else:
        return [s2tohc(s, k-1), s2tohc(-s, k-1)]

def sam2k_stoll(slst, mlst, counts):
    sdict = { mlst[i] : s2tohc(slst[i], len(mlst[i])) if type(mlst[i])==tuple else [slst[i], 1-slst[i]] for i in range(len(slst))}
    names, probs = samnk_prob(sdict)
    return samnk_loglikelihood(probs, counts)

def integrate1_range(nlst, llim):
    ntot = sum(nlst)
    if ntot==0: ntot=1
    n = nlst[0]
    lnp = lambda p: n*safelog(p) + (ntot-n)*safelog(1-p)
    e = lnp(n/ntot)
    plow = find_solution(lnp, n/ntot, 0.0, e-llim, 0.1)
    phigh = find_solution(lnp, n/ntot, 1.0, e-llim, 0.1)
    return [plow, phigh]

def integrate1_peak(nlst):
    return (nlst[0]+1)/(sum(nlst)+2)

def sam2k_logintegrate_1(s, counts, names, num):
    rngs = { n : (0.0, 1.0) for n in names }
    for k in range(2, len(names)):
        for t in hc_subsets(names, k):
            rngs[tuple(t)] = [-1.0, 1.0]
    mlst = [k for k in rngs] + [tuple(names)]
    res = [ sam2k_stoll([uniform(rngs[r][0], rngs[r][1]) for r in rngs] + [s], mlst, counts) for i in range(num)]
    mx = max(res)
    return log(sum([exp(x - mx) for x in res])) - log(num) + mx
    
def sam2k_logintegrate_2(s, counts, names, num):
    rngs = { names[i] : integrate1_range(hc_margin(counts, [i]), 14) for i in range(len(names)) }
    diff = sum([log(rngs[r][1]-rngs[r][0]) for r in rngs])
    for k in range(2, len(names)):
        for t in hc_subsets(names, k):
            rngs[tuple(t)] = [-1.0, 1.0]
    mlst = [k for k in rngs] + [tuple(names)]
    res = [ sam2k_stoll([uniform(rngs[r][0], rngs[r][1]) for r in rngs] + [s], mlst, counts) for i in range(num)]
    mx = max(res)
    return log(sum([exp(x - mx) for x in res])) - log(num) + diff + mx
    
def sam2k_logintegrate_3(srng, counts, names):
    rngs = { names[i] : integrate1_peak(hc_margin(counts, [i])) for i in range(len(names)) }
    #for k in range(2, len(names)):
    #    for t in hc_subsets(names, k):
    #        rngs[tuple(t)] = [-1.0, 1.0]
    mlst = [k for k in rngs] + [tuple(names)]
    res = [ sam2k_stoll([rngs[r] for r in rngs] + [s], mlst, counts) for s in srng]
    mx = max(res)
    sm = 0.0
    for i in range(len(res)):
        sm += exp(res[i] - mx)
    norm = safelog(sm) + mx
    return [srng, list(map(lambda x:exp(x-norm), res))]

def integratenk_log(s, nlst, rng):
    # Integrate (numerically) over all lower order interactions to find distribution over highest order
    pass


def sam2_avg(sam):
    if type(sam)==list:
        return (sam2_avg(sam[0])-sam2_avg(sam[1]))/2
    else:
        return sam

def sam_on_data(attrs):
    nd={}
    i=0
    for n in nodes:
        nd[n.name]=i
        i+=1
    lst = [nd[a] for a in attrs]
    lst.reverse()
    d = get_data(lst)
    res = samnk_from_prob(attrs, d)
    return { k : sam2_avg(res[k]) for k in res if type(k)==tuple}


def integrate2_point_pxpy_log(s, nlst, px, py):
    res=dotp(nlst, list(map(safelog,sam2_prob(s,px,py))))
    return res

def sam2_histogram_point(nlsts, px, py):
    sfunc = lambda s: sum(map(lambda nlst: integrate2_point_pxpy_log(s, nlst, px, py), nlsts))
    (srng,res) = find_calc_irange(sfunc, -1, 1, 0.2, 12)
    return [srng, normalize_logprob(res)]

def sam2k_histogram_point(ncounts):
    kk = depth(ncounts[0])
    mlst0 = list(range(kk))
    mlst = list(mlst0)
    for k in range(2, kk):
        for t in hc_subsets(mlst0, k):
            mlst += [tuple(t)]
    mlst += [tuple(mlst0)]
    params = []
    for counts in ncounts:
        par = [ integrate1_peak(hc_margin(counts, [i])) for i in mlst0 ]
        bcounts = hc_wsum(counts, hc_outer([[p, 1.0-p] for p in par]), 1.0, 1.0)
        for k in range(2, kk):
            for t in hc_subsets(mlst0, k):
                par += [sam2_avg(hc_lintrans(hc_uniippf(hc_margin(bcounts, list(t))), 1.0, -1.0))]
        params += [par]
    sfunc = lambda s: sum(map(lambda counts,par: sam2k_stoll(par + [s], mlst, counts), ncounts, params))
    (srng,res) = find_calc_irange(sfunc, -1, 1, 0.2, 12)
    return [srng, normalize_logprob(res)]

# Does not work
def sam2kb_histogram_point(ncounts, dims):
    kk = depth(ncounts[0])
    mlst0 = list(range(kk))
    mlst = list(mlst0)
    for k in range(2, kk):
        for t in hc_subsets(mlst0, k):
            mlst += [tuple(t)]
    mlst += [tuple(mlst0)]
    params = []
    for counts in ncounts:
        par = [ integrate_n1b_peak(hc_margin(counts, [i]), dims[i]) for i in mlst0 ]
        bcounts = hc_wsum(counts, hc_outer([[p, 1.0-p] for p in par]), 1.0, 1.0)
        for k in range(2, kk):
            for t in hc_subsets(mlst0, k):
                par += [sam2_avg(hc_lintrans(hc_uniippf(hc_margin(bcounts, list(t))), 1.0, -1.0))]
        params += [par]
    sfunc = lambda s: sum(map(lambda counts,par: sam2k_stoll(par + [s], mlst, counts), ncounts, params))
    (srng,res) = find_calc_irange(sfunc, -1, 1, 0.2, 12)
    return [srng, normalize_logprob(res)]

def samnk_histogram_point(ncounts):
    global debugdata
    kk = depth(ncounts[0])
    mlst0 = list(range(kk))
    mlst = list(mlst0)
    for k in range(2, kk):
        for t in hc_subsets(mlst0, k):
            mlst += [tuple(t)]
    mlst += [tuple(mlst0)]
    params = []
    ssum = hc_lintrans(ncounts[0], 0.0, 0.0)
    nsum = 0
    for counts in ncounts:
        par = [ integrate_n1_peak(hc_margin(counts, [i])) for i in mlst0 ]
        bcounts = hc_wsum(counts, hc_outer(par), 1.0, 1.0)
        for k in range(2, kk):
            for t in hc_subsets(mlst0, k):
                par += [hc_lintrans(hc_uniippf(hc_margin(bcounts, t)), 1.0, -1.0)]
        params += [par]
        n = hc_margin(bcounts, [])
        nsum += n
        ssum = hc_wsum(ssum, hc_lintrans(hc_uniippf(bcounts), 1.0, -1.0), 1.0, n)
    sfunc = lambda s: sum(map(lambda counts,par: samnk_stoll(par + [s], mlst, counts), ncounts, params))
    ssum = hc_lintrans(ssum, 1.0/nsum, 0.0)
    #
    # s_ij...k är en k-dimensionell lista, med värden ur en mångdimensionell simplex
    # hitta lämpligt integreringsintervall för var och en av s_ijk
    # antingen stega igenom hela intervallen, eller slumpa ut punkter
    # summera både individuella histogram för s_ijk och gemensamt för |0-s|
    #
    return (sfunc, ssum)
#    srng = samnk_simplexrange(sfunc, ssum)

def simplex_step(si, margs):
    # Om initialiserad, prova först att stega upp resten på samma nivå,
    #  annars, stega upp/initialisera första elementet rekursivt och sen prova resten,
    #  tills lyckas eller första elementet inte lyckas
    # Updatera margs till resten:
    #  om marg längs samma dim: subtrahera räknare från marg
    #  om längs annan dim: ta bort första elementet
    # Updatera margs till nästa nivå:
    #  om marg längs samma dim: lämna orörd
    #  om längs annan dim: ta bara första elementet
    
    ok = False
    if not simplex_is_reset(si):
        if simplex_step(simplex_rest(si)):
            ok = True
    while not ok:
        if not simplex_step(simplex_first(si)):
            return False
        if simplex_step(simplex_rest(si)):
            ok = True
    return si

def simplex1_step(si, bud, marg):
    if si[0]==-1:
        for i in reversed(range(len(si))):
            si[i] = min(marg[i], bud) if marg is not [] else bud
            bud -= si[i]
        if bud > 0:
            for i in range(len(si)):
                si[i] = -1
            return False
        return si
    else:
        ii = len(si)
        ok = False
        for i in range(len(si)):
            if si[i] == bud:
                ii = i
                break
            else:
                bud -= si[i]
        for i in reversed(range(ii)):
            if si[i] < marg[i]:
                si[i] += 1
                bud -= 1
                ii = i+1
                ok = True
                break
            else:
                bud += si[i]
        if not ok:
            for i in range(len(si)):
                si[i] = -1
            return False
        for i in reversed(range(ii, len(si))):
            si[i] = min(marg[i], bud) if marg is not [] else bud
            bud -= si[i]
        return si

def simplex2_step(si, bud, marg):
    ok = False
    if not si[0][0] == -1:
        if len(si)==1:
            if simplex1_step(si[0], marg[0], bud):
                ok = True
            else:
                return False
        elif simplex2_step(si[1:], list(map(lambda x,y:x-y, bud, si[0])), marg[1:]):
            ok = True
    while not ok:
        if not simplex1_step(si[0], marg[0], bud):
            return False
        if len(si)==1 or simplex2_step(si[1:], list(map(lambda x,y:x-y, bud, si[0])), marg[1:]):
            ok = True
    return si

def simplex_range(s, stepf):
    dim = dims(s)
    n = dimsprod(s)*stepf
    delta = -hc_margin(s, []) / n
    si = make_hc(dim, -1)
    # Assume 2-dimensional for now
    bud = make_hc([dim[1]], n/dim[1])
    marg = make_hc([dim[0]], n/dim[0])
    while simplex2_step(si, bud, marg) is not False:
        yield hc_wsum(s, si, 1.0, delta)

def metaind_range(dim):
    ind = [0]*len(dim)
    ok = True
    ind[-1] = -1
    while ok:
        ok = False
        for i in reversed(range(len(dim))):
            if ind[i] < dim[i]-1:
                ind[i] += 1
                ok = True
                yield ind.copy()
                break
            else:
                ind[i] = 0

#
# Now we will go for a Gibbs sampling algorithm.
# First find the peak by starting from 0, iteratively draw a random ind, and find
# low and high and peak.
# Then from the peak, iteratively draw a random ind, find its histogram, add to accumulators.
#
def samnk_significance_gibbs(ncounts, tol):
    def random_ind(dim, last):
        val = int(random()*dim)
        return val if val < last else val+1
    dim = dims(ncounts[0])
    kk = depth(ncounts[0])
    lim = dimsprod(ncounts[0])*8
    mlst0 = list(range(kk))
    mlst = list(mlst0)
    for k in range(2, kk):
        for t in hc_subsets(mlst0, k):
            mlst += [tuple(t)]
    mlst += [tuple(mlst0)]
    params = []
    for counts in ncounts:
        par = [ integrate_n1_peak(hc_margin(counts, [i])) for i in mlst0 ]
        bcounts = hc_wsum(counts, hc_outer(par), 1.0, 1.0)
        for k in range(2, kk):
            for t in hc_subsets(mlst0, k):
                par += [hc_lintrans(hc_uniippf(hc_margin(bcounts, t)), 1.0, -1.0)]
        params += [par]
    sfunc = lambda s: sum(map(lambda counts,par: samnk_stoll(par + [s], mlst, counts), ncounts, params))
    s = make_hc(dim)
    lastind = dim
    stable = 0
    bests = s
    bestll = sfunc(s)
    while stable < 8:
        ind = [random_ind(d, l) for (d,l) in zip(dim, lastind)]
        low = hc_simplexlower(s, ind)
        high = hc_simplexheighten(s, ind)
        (x, ll) = find_peak(lambda x: sfunc(hc_wsum(low, high, 1.0-x, x)), 0, 1, 0.5)
        s = hc_wsum(low, high, 1.0-x, x)
        if ll > bestll:
            bestll = ll
            bests = s
            stable = 0
        else:
            stable += 1
    cnt = 0
    norm = sqrt(hc_dotp(bests, bests))
    pzero = [0.0, 0.0]
    avg = 0.0
    s = bests
    while cnt < lim:
        ind = [random_ind(d, l) for (d,l) in zip(dim, lastind)]
        low = hc_simplexlower(s, ind)
        high = hc_simplexheighten(s, ind)
        (xres, yres) = find_calc_irange(lambda x: sfunc(hc_wsum(low, high, 1.0-x, x)), 0, 1, 0.5, 14)
        yres = normalize_logprob(yres)
        u = random()
        acc = 0.0
        for (x,y) in zip(xres,yres):
            acc += y
            if acc >= u:
                s = hc_wsum(low, high, 1.0-x, x)
                break
        projlow = hc_dotp(bests, low)/norm
        projhigh = hc_dotp(bests, high)/norm
        xres = [(1-x)*projlow + x*projhigh for x in xres]
        mean,var = hist_mean_var([xres,yres])
        avg += mean
        for (x,y) in zip(xres,yres):
            pzero[0 if np.sign(x) != np.sign(mean) else 1] += y
        cnt += 1
    return (pzero[0]/cnt, avg/cnt)

def samnk_significance_gibbs_1(ncounts):
    kk = depth(ncounts[0])
    mlst0 = list(range(kk))
    mlst = list(mlst0)
    for k in range(2, kk):
        for t in hc_subsets(mlst0, k):
            mlst += [tuple(t)]
    mlst += [tuple(mlst0)]
    params = []
    for counts in ncounts:
        par = [ integrate_n1_peak(hc_margin(counts, [i])) for i in mlst0 ]
        bcounts = hc_wsum(counts, hc_outer(par), 1.0, 1.0)
        for k in range(2, kk):
            for t in hc_subsets(mlst0, k):
                par += [hc_lintrans(hc_uniippf(hc_margin(bcounts, t)), 1.0, -1.0)]
        params += [par]
    sfunc = lambda s: sum(map(lambda counts,par: samnk_stoll(par + [s], mlst, counts), ncounts, params))
    return sfunc

def samnk_significance_gibbs_2(sfunc, dim):
    #dim = dims(ncounts[0])
    #lim = dimsprod(ncounts[0])*8
    def random_ind(dim, last):
        val = int(random()*dim)
        return val if val < last else val+1
    s = make_hc(dim)
    lastind = dim
    stable = 0
    bests = s
    bestll = sfunc(s)
    while stable < 8:
        ind = [random_ind(d, l) for (d,l) in zip(dim, lastind)]
        low = hc_simplexlower(s, ind)
        high = hc_simplexheighten(s, ind)
        (x, ll) = find_peak(lambda x: sfunc(hc_wsum(low, high, 1.0-x, x)), 0, 1, 0.5)
        s = hc_wsum(low, high, 1.0-x, x)
        if ll > bestll:
            bestll = ll
            bests = s
            stable = 0
        else:
            stable += 1
    return bests

def samnk_significance_gibbs_3(sfunc, dim, bests, lim):
    def random_ind(dim, last):
        val = int(random()*dim)
        return val if val < last else val+1
    cnt = 0
    norm = sqrt(hc_dotp(bests, bests))
    pzero = [0.0, 0.0]
    avg = 0.0
    lastind = dim
    s = bests
    while cnt < lim:
        ind = [random_ind(d, l) for (d,l) in zip(dim, lastind)]
        low = hc_simplexlower(s, ind)
        high = hc_simplexheighten(s, ind)
        (xres, yres) = find_calc_irange(lambda x: sfunc(hc_wsum(low, high, 1.0-x, x)), 0, 1, 0.5, 14)
        yres = normalize_logprob(yres)
        u = random()
        acc = 0.0
        for (x,y) in zip(xres,yres):
            acc += y
            if acc >= u:
                s = hc_wsum(low, high, 1.0-x, x)
                break
        projlow = hc_dotp(bests, low) / norm
        projhigh = hc_dotp(bests, high) / norm
        xres = [(1-x)*projlow + x*projhigh for x in xres]
        mean,var = hist_mean_var([xres,yres])
        avg += mean
        for (x,y) in zip(xres,yres):
            pzero[0 if np.sign(x) != np.sign(mean) else 1] += y
        cnt += 1
    return (pzero[0]/cnt, avg/cnt)

def samnk_significance_gibbs_3B(sfunc, dim, bests):
    norm = sqrt(hc_dotp(bests, bests))
    intrdim = 1
    for k in dim:
        intrdim *= (k-1)
    pzero = [0.0, 0.0]
    avg = 0.0
    cnt = 0.0
    for ind in metaind_range(dim):
        low = hc_simplexlower(bests, ind)
        high = hc_simplexheighten(bests, ind)
        (xres, yres) = find_calc_irange(lambda x: sfunc(hc_wsum(low, high, 1.0-x, x)), 0, 1, 0.5, 14)
        # vikta om yres beroende på radie och intrinsisk dimension
        yres = [yres[i] + safelog(abs(xres[i]-norm))*(intrdim-1) for i in range(len(xres))]
        yres = normalize_logprob(yres)
        projlow = hc_dotp(bests, low) / norm
        projhigh = hc_dotp(bests, high) / norm
        xres = [(1-x)*projlow + x*projhigh for x in xres]
        mean,var = hist_mean_var([xres,yres])
        avg += mean
        for (x,y) in zip(xres,yres):
            pzero[0 if np.sign(x) != np.sign(mean) else 1] += y
        cnt += 1
    return (pzero[0]/cnt, avg/cnt)

def samnk_significance_gibbs_3C(sfunc, dim, bests, fact):
    def interval_union(i1, i2):
        if i1 is False:
            return [min(i2), max(i2)]
        else:
            return [min([i1[0]] + i2), max([i1[1]] + i2)]
    norm = sqrt(hc_dotp(bests, bests))
    slow = make_hc(dim)
    ival = False
    for ind in metaind_range(dim):
        low = hc_simplexlower(bests, ind)
        high = hc_simplexheighten(bests, ind)
        (a, b, m, delta) = find_irange(lambda x: sfunc(hc_wsum(low, high, 1.0-x, x)), 0, 1, 0.5, 14)
        projlow = hc_dotp(bests, low) / norm
        projhigh = hc_dotp(bests, high) / norm
        ival = interval_union(ival, [(1-x)*projlow + x*projhigh for x in [a, b]])
        hc_setpart(slow, ind, (1-a)*hc_part(low, ind) + a*hc_part(high, ind))
    pzero = [0.0, 0.0]
    avg = 0.0
    xres = []
    yres = []
    for s in simplex_range(slow, fact):
        yres.append(sfunc(s))
        xres.append(hc_dotp(bests, s) / norm)
    yres = normalize_logprob(yres)
    avg,var = hist_mean_var([xres,yres])
    for (x,y) in zip(xres,yres):
        pzero[0 if np.sign(x) == -1 else 1] += y
    # histogram
    return (pzero, avg)

# Does not work
def samnk_significance_point(ncounts, tol = 0.0):
    # Eller så gör vi tvärtom och hittar individuella signifikanser för varje s_ijk med
    # hjälp av den kod som redan finns.
    #
    bestsg = 1.0
    bestmean = 0.0
    bestvar = 0.0
    besthist = False
    dms = dims(ncounts[0])
    for inds in hc_range_no2(dms):
        ncounts2 = [hc_binarize(counts, inds) for counts in ncounts]
        h = sam2kb_histogram_point(ncounts2, dms)
        mean,var = hist_mean_var(h)
        sg = hist_significance(h, 0.0, mean, tol)
        if sg < bestsg or sg == bestsg and mean > bestmean:
            bestsg = sg
            bestmean = mean
            bestvar = var
            besthist = h
    return (bestsg, bestmean, bestvar, besthist)

#        par2 = [[hc_part(p, listpart(inds, m)) for (p, m) in zip(par, mlst[:-1])] for par in params]
#        ncounts2 = [hc_binarize(counts, inds) for counts in ncounts]
#        debugdata = (ncounts2, inds, par2)
#        sfunc = lambda s: sum(map(lambda counts,par: sam2k_stoll(par + [s], mlst, counts), ncounts2, par2))
#        srng = list(sam2_srange(sfunc))
#        res = [sfunc(s) for s in srng]
#        mx = max(res)
#        sm = 0.0
#        for i in range(len(res)):
#            sm += exp(res[i] - mx)
#        norm = safelog(sm) + mx
#        h = [srng, list(map(lambda x:exp(x-norm), res))]

# Does not work
def sam2kb_histogram_point_alt(ncounts, params, mlst, inds, dms):
    # ****
    params2 = [[listpart(p,[inds[m] for m in marg]) for p,marg in zip(par,mlst)] for par in params]
    sfunc = lambda s: sum(map(lambda counts,par: sam2kb_stoll(par + [s], mlst, counts, dms), ncounts, params2))
    (srng,res) = find_calc_irange(sfunc, -1, 1, 0.2, 12)
    return [srng, normalize_logprob(res)]

# Does not work
def samnk_significance_point_alt(ncounts, tol = 0.0):
    # Eller så gör vi tvärtom och hittar individuella signifikanser för varje s_ijk med
    # hjälp av den kod som redan finns.
    bestsg = 1.0
    bestmean = 0.0
    bestvar = 0.0
    besthist = False
    dms = dims(ncounts[0])
    kk = depth(ncounts[0])
    mlst0 = list(range(kk))
    mlst = list(mlst0)
    for k in range(2, kk):
        for t in hc_subsets(mlst0, k):
            mlst += [tuple(t)]
    mlst += [tuple(mlst0)]
    params = []
    for counts in ncounts:
        par = [ integrate_n1_peak(hc_margin(counts, [i])) for i in mlst0 ]
        bcounts = hc_wsum(counts, hc_outer(par), 1.0, 1.0)
        for k in range(2, kk):
            for t in hc_subsets(mlst0, k):
                par += [sam2_avg(hc_lintrans(hc_uniippf(hc_margin(bcounts, list(t))), 1.0, -1.0))]
        params += [par]
    for inds in hc_range_no2(dms):
        h = sam2kb_histogram_point_alt(ncounts, params, mlst, inds, dms)
        mean,var = hist_mean_var(h)
        sg = hist_significance(h, 0.0, mean, tol)
        if sg < bestsg or sg == bestsg and mean > bestmean:
            bestsg = sg
            bestmean = mean
            bestvar = var
            besthist = h
    return (bestsg, bestmean, bestvar, besthist)

def samnk_signif(data, attrs, cond, tol=0.0):
    d = get_nhc_data(data[1], cond + attrs, [data[0][i][1] for i in cond + attrs])
    v = nkflat(d, len(attrs))
    return samnk_significance_gibbs(v, tol) + (False,)
#    return samnk_significance_point(v) + (False,)

def samnk_signif_general(data, form, attrs, cond, tol, pseudo = False):
    # if cond contains no continuous, simple case
    # otherwise try first level
    # if non-significant stop
    # otherwise try second level
    # if moves away stop
    # otherwise try third level
    # if converges, calc goal, if moves away stop
    # otherwise try fourth
    # if converges, calc goal, else use third
    #func = lambda lev: samnk_significance_point(nkflat(get_general_hc_data(data, form, attrs, cond, pseudo, lev), len(attrs)), tol)
    func = lambda lev: samnk_significance_gibbs(nkflat(get_general_hc_data(data, form, attrs, cond, pseudo, lev), len(attrs)), tol)
    cont = False
    for c in cond:
        if form[c][0]=='cont':
            cont = True
            break
    if not cont:
        #(sig, mean, var, h) = func(0)
        (sig, mean) = func(0)
        return (sig, mean, False)
    else:
        print(attrs, cond)
        (sig1, mean1, var1, h1) = func(0)
        (sig2, mean2, var2, h2) = func(1)
        #(sig3, mean3, var3, h3) = func(2)
        #if mean2 <= tol and mean2 >= -tol:
        #    return (sig2, mean2, h2)
        if abs(mean2) >= abs(mean1):
            return (sig2, mean2, h2)
        (sig3, mean3, var3, h3) = func(2)
        print([mean1, mean2, mean3],  " -> ", mean2 + (mean3 - mean2)*(mean2 - mean1)/((mean2 - mean1) - (mean3 - mean2)), [var1, var2, var3], [sig1, sig2, sig3])
        if abs(mean3) >= abs(mean2):
            return (sig3, mean3, h3)
        if abs(mean3 - mean2) < abs(mean2 - mean1)*0.8:
            dmean = (mean3 - mean2)*(mean3 - mean2)/((mean2 - mean1) - (mean3 - mean2))
            nh = [[x+dmean for x in h3[0]], h3[1]]
            nmean = mean3 + dmean
            nsig = hist_significance(nh, 0.0, nmean, tol)
            print("L3", nsig, nmean, dmean) 
            return (nsig, nmean, nh)
        else:
            return (sig3, mean3, h3)

def samnk_signif_general_basic(data, form, attrs, cond, tol, pseudo = False):
    #func = lambda lev: samnk_significance_point(nkflat(get_general_hc_data(data, form, attrs, cond, pseudo, lev), len(attrs)), tol)
    func = lambda lev: samnk_significance_gibbs(nkflat(get_general_hc_data(data, form, attrs, cond, pseudo, lev), len(attrs)), tol)
    (sig, mean, var, h) = func(1)
    return (sig, mean, h)

#        (sig4, mean4, var4, h4) = func(3)
#        print([mean2, mean3, mean4], [var2, var3, var4], [sig2, sig3, sig4])
#        if abs(mean4) >= abs(mean4):
#            return (sig3, mean3, h3)
#        if abs(mean4 - mean3) < abs(mean3 - mean2)*0.8:
#            dmean = (mean4 - mean3)/(1 - (mean4 - mean3)/(mean3 - mean2))
#            nh = [[x+dmean for x in h4[0]], h4[1]]
#            nmean = mean4 + dmean
#            nsig = hist_significance(nh, 0.0, nmean, tol)
#            print("L4", nsig, nmean, dmean) 
#            return (nsig, nmean, nh)

def samnk_signif_general_bad(data, form, attrs, cond, tol, pseudo = False):
    global discretization
    func = lambda i,lev: samnk_significance_point(nkflat(get_general_hc_data_alt(data, form, cond + attrs, [discretization<<(lev if i==j else 1) for j in range(len(cond))] + [discretization]*len(attrs), pseudo), len(attrs)), tol)
    dmean = 0.0
    (sig2, mean2, var2, h2) = func(-1,1)
    for i,c in enumerate(cond):
        if form[c][0]=='cont':
            (sig1, mean1, var1, h1) = func(i,0)
            (sig3, mean3, var3, h3) = func(i,2)
            if np.sign(mean3 - mean2) == np.sign(mean2 - mean1) and abs(mean3 - mean2) < abs(mean2 - mean1)*0.8:
                dmean += (mean3 - mean2)*(mean2 - mean1)/((mean2 - mean1) - (mean3 - mean2))
                print(c, [mean1, mean2, mean3], " -> ", mean2 + (mean3 - mean2)*(mean2 - mean1)/((mean2 - mean1) - (mean3 - mean2)))
            else:
                dmean += (mean3 - mean2)
                print(c, [mean1, mean2, mean3], " == ", mean3)
    if dmean == 0.0:
        return (sig2, mean2, h2)
    else:
        nh = [[x+dmean for x in h2[0]], h2[1]]
        nmean = mean2 + dmean
        nsig = hist_significance(nh, 0.0, nmean, tol)
        print("Total ", nsig, nmean, dmean) 
        return (nsig, nmean, nh)

def sam2k_signif(data, attrs, cond):
    d = get_hc_data(data, cond + attrs)
    v = kflat(d, len(attrs))
    h = sam2k_histogram_point(v)
    mean,var = hist_mean_var(h)
    sg = hist_significance(h, 0.0, mean)
    return (sg, mean, h)

def sam2_signif(data, attrs, cond):
    d = get_hc_data(data, cond + attrs)
    v = twoflat(d)
    h = sam2_histogram(v)
    mean,var = hist_mean_var(h)
    return (hist_significance(h,0.0,mean), mean, h)


#-------------------
# Gaussian tests

def matrix_inverse(mat):
    return list(map(list, list(np.linalg.inv(mat))))

def matrix_select(mat, attrs):
    return [[mat[i][j] for i in attrs] for j in attrs]

def gauss_tail(std):
    return erfc(std / sqrt(2.0)) / 2.0

def gauss_signif(data, attrs, cond, tol):
    num = len(data)
    cov = get_covar_data(data, attrs + cond)
    inv = matrix_inverse(cov)
    r = -inv[0][1]/sqrt(inv[0][0]*inv[1][1]) if inv[0][0] > 0.0 and inv[1][1] > 0.0 else 0.0
    if tol != 0.0:
        if r > 0.0:
            r = max(0.0, r - tol)
        else:
            r = min(0.0, r + tol)
    sg = (gauss_tail(0.5*abs(log((1.0 + r)/(1.0 - r)))*sqrt(num-3.0)) if abs(r) < 1.0 else 0.0) if num > 3.0 else 1.0 
    return (sg, r)

def gauss3_signif(data, attrs, cond):
    num = len(data)
    r = get_trivar_cond_data(data, attrs, cond)
    # *** test below may not be accurate
    sg = (gauss_tail(0.5*abs(log((1.0 + r)/(1.0 - r)))*sqrt(num-3.0)) if abs(r) < 1.0 else 0.0) if num > 3.0 else 1.0 
    return (sg, r)

def gauss_bin_signif(data, attrs, cond):
    num = len(data)
    alst = cond + attrs
    meanvec = get_mean_data(data, alst)
    cube = 0
    for ind in alst:
        cube=[cube, deepcopy(cube)]
    for vec in data:
        incrlistpart(cube, list(map(lambda i:vec[alst[i]]>meanvec[i], range(len(alst)))))
    v = twoflat(cube)
    h = sam2_histogram(v)
    mean,var = hist_mean_var(h)
    return (hist_significance(h,0.0,mean), mean, h)
    
def gauss_bink_signif(data, attrs, cond):
    num = len(data)
    alst = cond + attrs
    meanvec = get_mean_data(data, alst)
    cube = 0
    for ind in alst:
        cube=[cube, deepcopy(cube)]
    for vec in data:
        incrlistpart(cube, list(map(lambda i:vec[alst[i]]>meanvec[i], range(len(alst)))))
    v = kflat(cube, len(attrs))
    h = sam2k_histogram_point(v)
    mean,var = hist_mean_var(h)
    return (hist_significance(h,0.0,mean), mean, h)
    
# (a b)-1 = (c -b)
# (b c)     (-b a)/(ac-b2)
#
# r = b/sqrt(ac) = -(-b/(ac-b2))/sqrt(ca/(ac-b2)^2)
#
# 1/2 log((1+r)/(1-r)) = 1/2 log((sqrt(ac)+b)/(sqrt(ac)-b))


#-------------------
# Entry points

def distr(data, attr, stype, form = None):
    if stype in ['sam', 'fisher', 'g2'] or form is not None and form[attr][0]=='bin':
        d = get_hc_data(data, [attr])
        pr = integrate1_peak(d) #(d[0]+1.0)/(d[0]+d[1]+2.0)
        func = lambda p: safelog(p)*d[0] + safelog(1-p)*d[1]
        x = [0.025*i for i in range(41)]
        y = [func(p) for p in x]
        maxy = max(y)
        y = [exp(v-maxy) for v in y]
        return (pr, [x,y])
    elif stype in ['gauss', 'gaussbin'] or form is not None and form[attr][0]=='cont':
        mean = get_mean_data(data, [attr])[0]
        std = sqrt(get_covar_data(data, [attr])[0][0])
        x = [(i-30)/30.0 for i in range(61)]
        y = [0.0 for i in range(61)]
        for vec in data:
            val = (vec[attr] - mean)/std if std > 0 else 0.0
            ind = min(30, max(-30, int(round(val*7.5))))
            y[ind+30] += 1.0
        maxy = max(y)
        if maxy > 0.0:
            y = [v/maxy for v in y]
        return (False, [x,y])
    else:
        return (False, False)

def signif(data, attrs, cond, stype, form = None, tol = 0.0):
    # Returns significance, mean, and if possible histogram
    if (stype == 'sam'):
        return sam2_signif(data, attrs, cond)
    elif (stype == 'nsam'):
        return samnk_signif(data, attrs, cond)
    elif (stype == 'gsam'):
        return samnk_signif_general(data, form, attrs, cond, tol, False)
    elif (stype == 'psam'):
        return samnk_signif_general(data, form, attrs, cond, tol, True)
    elif (stype == 'fisher'):
        return fisher_signif(data, attrs, cond) + (False,)
    elif (stype == 'g2'):
        return g2_signif(data, attrs, cond) + (False,)
    elif (stype == 'gauss'):
        return gauss_signif(data, attrs, cond, tol) + (False, )
    elif (stype == 'gaussbin'):
        return gauss_bin_signif(data, attrs, cond)
    else:
        return (1.0, 0.0, False)

def trisignif(data, attrs, cond, stype, form = None, tol = 0.0):
    if (stype == 'sam'):
        return sam2k_signif(data, attrs, cond)
    elif (stype == 'nsam'):
        return samnk_signif(data, attrs, cond)
    elif (stype == 'bsam'):
        return samnk_signif_general_basic(data, form, attrs, cond, tol, False)
    elif (stype == 'gsam'):
        return samnk_signif_general(data, form, attrs, cond, tol, False)
    elif (stype == 'psam'):
        return samnk_signif_general(data, form, attrs, cond, tol, True)
    elif (stype == 'gauss'):
        return gauss3_signif(data, attrs, cond) + (False, )
    elif (stype == 'gaussbin'):
        return gauss_bink_signif(data, attrs, cond)
    else:
        return (1.0, 0.0, False)
    
