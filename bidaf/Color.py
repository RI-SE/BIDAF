import numpy as np

def colorgamma(x):
    if x < 0.0031308:
        return x*12.92
    else:
        return 1.055 * np.power(x, 0.416667) - 0.055

def invcolorgamma(x):
    if x < 0.04045:
        return x / 12.92
    else:
        return np.power((x + 0.055)/1.055, 2.4)

def rgb_color(r, g, b):
    return tuple(list(map(colorgamma, [r,g,b])))

# mimics color.scm with 6color, bluebased, and smooth
def hsl_color(h, s, l):
    hramp = [0, 1/12, 1/6, 1/3, 2/3, 7/9, 1]
    iramp = [(2.410996, 0.16, 0, 0, 1), (0.862552, 0.79, 0, 1, 1),
             (-0.252442, 0.63, 0, 1, 0), (-1.981885, 0.84, 1, 1, 0),
             (1.786451, 0.21, 1, 0, 0), (1.571190, 0.37, 1, 0, 1),
             (2.410996, 0.16, 0, 0, 1)]
    i = 0
    while h > hramp[i+1]:
        i += 1
    p = (h - hramp[i])/(hramp[i+1] - hramp[i])
    (a, br, r, g, b) = tuple(map(lambda x1, x2: p*(x2 - x1) + x1, iramp[i], iramp[i+1]))
    ll0 = (l + 1.0)*0.5
    ll = (np.exp(a*ll0) - 1.0)/(np.exp(a) - 1.0) if not a==0 else ll0
    if ll < br:
        t1 = s * ll / br
        t2 = ll0 * (1.0 - s)
    else:
        t1 = s * (1.0 - ll) / (1.0 - br)
        t2 = ll0 * (1.0 - s) + s * (ll - br) / (1.0 - br)
    return rgb_color(r*t1+t2, g*t1+t2, b*t1+t2)

def gencolor(light=0.0, starthue=0.0, startsat=1.0, unsat=False):
    h = starthue
    s = startsat
    if unsat:
        ss = s*s
    l = light
    gs1 = (3.0 - np.sqrt(5))/2.0
    gs2 = gs1/np.sqrt(3)
    while True:
        yield hsl_color(h, s, l)
        h -= gs1
        if h < 0.0:
            h += 1.0
        if unsat:
            ss -= gs2
            if ss <= 0:
                ss += 1.0
            s = np.sqrt(ss)

