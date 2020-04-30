import numpy as np
import numpy.testing as npt

from cvdm.score import cox_surv, weibull_atf_surv, weibull_hazard, weibull_surv


def test_cox_surv():
    beta = np.array([ 2.32888,
                      1.20904,
                     -0.70833,
                      2.76157,
                      2.82263,
                      0.52873,
                      0.69154])
    s0 = 0.95012
    const = 26.1931
    xFeat = np.array([np.log(61),
                      np.log(180), 
                      np.log(47), 
                      np.log(124)*1, 
                      np.log(124)*0,
                      1, 
                      0])
    tmp = cox_surv(xFeat, beta, s0, const)
    npt.assert_almost_equal(tmp, 0.1048, decimal=4)


def test_weibull_hazard():
    tmp = weibull_hazard(np.array([62, 30, 32, 
                                   5, 1, 1]),
                         np.array([0.068, 0.012, 0.072,
                                  -0.22, 0.771, 0.658]),
                        -12.332,
                         8,
                         1.514)
    npt.assert_almost_equal(tmp, 0.1388, decimal=4)
    tmp = weibull_hazard(np.array([62, 30, 32, 
                                   5, 1, 1]),
                         np.array([0.068, 0.012, 0.072,
                                  -0.22, 0.771, 0.658]),
                        -12.332,
                         9,
                         1.514)
    npt.assert_almost_equal(tmp, 0.1659, decimal=4)


def test_weibull_surv():
    tmp = weibull_surv(np.array([62, 30, 32, 
                                   5, 1, 1]),
                         np.array([0.068, 0.012, 0.072,
                                  -0.22, 0.771, 0.658]),
                        -12.332,
                         8,
                         9,
                         1.514)
    npt.assert_almost_equal(tmp, 0.027, decimal=3)



def test_weibull_atf_surv():
    alpha = np.array([-0.287,
                      -0.026,   
                      -0.149,  
                       0.011,   
                      -0.268, 
                      -0.308,  
                       0.438,  
                      -0.712,  
                      -0.010,  
                      -1.292,  
                       0.009,  
                       1.241])
    xFeat = np.array([np.log(6),
                      59,
                      5.8,
                      0,
                      1,
                      1,
                      np.log(8),
                      np.log(8),
                      160,
                      0,
                      0,
                      1.7])
    mu = 11.262
    sigma = 0.587
    tmp = weibull_atf_surv(xFeat, alpha, mu, sigma, 5)
    npt.assert_almost_equal(tmp, 0.54, decimal=2)

