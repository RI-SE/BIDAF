from re import *

def sortunique(lst):
    last = None
    res = []
    for x in sorted(lst):
        if last != x:
            last = x
            res.append(x)
    return res

def split_feature(name):
    parts = split("_", name)
    mt = match("([a-zA-Z]+)([.0-9]+)",parts[-1])
    if len(parts) == 1 or not mt or len(mt.group(2))==0:
        return (name, None, None)
    return ("_".join(parts[:-1]), mt.group(1), eval(mt.group(2)))

def feature_base(name):
    return split_feature(name)[0]

def feature_mode(name):
    return split_feature(name)[1]

def feature_scale(name):
    return split_feature(name)[2]

def all_feature_bases(dicvec):
    return sortunique([feature_base(name) for name in dicvec if name not in ['time','entity']])

def all_feature_modes(dicvec):
    return sortunique([feature_mode(name) for name in dicvec if name not in ['time','entity']])

def all_feature_scales(dicvec):
    return sortunique([feature_scale(name) for name in dicvec if name not in ['time','entity']])

def filter_feature_base(dicvec, vals):
    return { name:dicvec[name] for name in dicvec if name in ['time','entity'] or feature_base(name) in vals}

def filter_feature_mode(dicvec, vals):
    return { name:dicvec[name] for name in dicvec if name in ['time','entity'] or feature_mode(name) in vals}

def filter_feature_scale(dicvec, vals):
    return { name:dicvec[name] for name in dicvec if name in ['time','entity'] or feature_scale(name) in vals}

def filter_names_base(colnames, vals):
    return [ name for name in colnames if name in ['time','entity'] or feature_base(name) in vals]

def filter_names_mode(colnames, vals):
    return [ name for name in colnames if name in ['time','entity'] or feature_mode(name) in vals]

def filter_names_scale(colnames, vals):
    return [ name for name in colnames if name in ['time','entity'] or feature_scale(name) in vals]

