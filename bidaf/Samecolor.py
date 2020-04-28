# två listor, med gamla och nya klusterindex.
# Varje (gammalt) index har en färg associerad med sig.
# Associera färger till nya index så att så många punkter som möjligt får samma färg
# Gör det genom att beräkna matris ellan gamla och nya, välj största talet,
# bestäm färg, sn rekursivt på resterande rader o columner.

import numpy as np

def samecolor(l1, l2, cols, cfunc=lambda x:x):
    newcols = {}
    taken = []
    m1 = max(l1)+1
    m2 = max(l2)+1
    matr = np.mat(np.zeros((m1,m2)))
    for i in range(min(len(l1),len(l2))):
        matr[l1[i],l2[i]] += 1
    # så länge rader: hitta max, nolla samma rad o kolumn
    for i in range(min(m1,m2)):
        ind = np.argmax(matr)
        (i1, i2) = (int(ind/m2), ind%m2)
        newcols[i2] = cols[i1]
        taken += [cols[i1]]
        for j in range(m1):
            matr[j, i2] = -1
        for j in range(m2):
            matr[i1, j] = -1
    i = 0
    for c in range(max(m1,m2,max(cols))):
        if not (c in cols and cols[c] in taken):
            while i in newcols:
                i += 1
            newcols[i] = cols[c] if c in cols else cfunc(c)
            i += 1
    return newcols


