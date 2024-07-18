import numpy as np
import time


def Levy(row, col, beta):
    num = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = sigma_u ** 2 * np.random.random(size=(row, col))
    v = np.random.random(size=(row, col))
    z = u / abs(v) ** (1 / beta)
    return z


def SSA(x, objfun, lb, ub, max_iter):
    Mini = lb[1, :]
    Maxi = ub[1, :]
    N, dim = x.shape[0], x.shape[1]
    ct = time.time()
    Convergence = np.zeros(max_iter)
    fitness = np.zeros(dim)

    fitness = objfun(x)
    index = np.where(fitness == np.min(fitness))[0][0]
    Bestfit = fitness[index]
    Bestsol = x[index, :]
    pdp = 0.1
    row = 1.204
    V = 5.25
    S = 0.0154
    cd = 0.6
    CL = 0.7
    hg = 1
    sf = 18
    Gc = 1.9
    D1 = 1 / (2 * row * V ** 2 * S * cd)
    L = 1 / (2 * row * V ** 2 * S * CL)
    tanpi = D1 / L
    dg = hg / (tanpi * sf)

    for iter in range(max_iter):
        for i in range(N):
            if np.random.random() >= pdp:
                x[i, :] = np.round(x[i, :] + dg * Gc * np.abs(Bestsol - x[i, :]))
            else:
                x[i, :] = Mini + (Maxi - Mini) * np.random.random(size=dim)
            Fh = x
            fitness[i] = objfun(x[i, :])
            ind1 = np.where(fitness == np.min(fitness))[0][0]
            Bestsol = x[ind1, :]
            if np.random.random() > pdp:
                x[i, :] = np.round(x[i, :] + dg * Gc * np.abs(Bestsol - x[i, :]))
            else:
                x[i, :] = Mini + (Maxi - Mini) * np.random.random(size=dim)
            Fa = x
            fitness[i] = objfun(x[i, :])
            ind2 = np.where(fitness == np.min(fitness))[0][0]
            Bestsol = x[ind2, :]

            fitness = objfun(x[i, :])

        Sc = np.sqrt(np.sum(np.abs(Fh - Fa), axis=0) ** 2)
        Smin = ((10 * np.exp(-6)) / 365) ** (iter / (max_iter / 2.5))
        if Sc[0] < Smin:
            season = 'summer'
            for i in range(N):
                x[i, :] = Mini + (Maxi - Mini) * Levy(1, dim, 1.5)
        else:
            season = 'winter'

        ind3 = np.where(fitness == np.min(fitness))[0][0]
        Bestsol = x[ind3, :]
        Final = np.zeros((3, dim))
        Final[0, :] = x[ind1, :]
        Final[1, :] = x[ind2, :]
        Final[2, :] = x[ind3, :]

        fitt = objfun(Final)

        index = np.where(fitt == np.min(fitt))[0][0]
        Bestsol = Final[index, :]
        Bestfit = fitt[index]
        Convergence[iter] = Bestfit
    ct = time.time() - ct
    return Bestfit, Convergence, Bestsol, ct
