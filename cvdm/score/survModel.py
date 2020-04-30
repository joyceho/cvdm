import numpy as np


def cox_surv(xFeat, beta, s0, b0=0, shrinkage=1):
    # 1 - x**exp(sum(xb))
    xBeta = xFeat.dot(beta)
    return 1 - s0**np.exp(shrinkage*(xBeta - b0))


def weibull_atf_surv(xFeat, beta, mu, sigma, t):
    a = xFeat.dot(beta) + mu
    b = (t / a) ** sigma
    return 1 - np.exp(-b)


def weibull_hazard(xFeat, beta, lmbda, t, rho):
    # exp(lambda + beta * xFeat) * t^rho
    return np.exp(lmbda + xFeat.dot(beta)) * t ** rho


def weibull_surv(xFeat, beta, lmbda, t1, t2, rho):
    return 1 - np.exp(weibull_hazard(xFeat, beta, lmbda, t1, rho) - weibull_hazard(xFeat, beta, lmbda, t2, rho))

