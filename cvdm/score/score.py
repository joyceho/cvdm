"""
SCORE Risk Calculation

See:
Conroy et al. (2003). Estimation of ten-year risk of fatal cardiovascular
disease in Europe: the SCORE project. European Heart Journal, 24(11), 987–1003.
"""
import numpy as np

from cvdm.score import BaseRisk, clean_chol, clean_bp


# coefficients for survival
LOW_RISK_MEN = {'CHD': {"alpha": -22.1, "p": 4.71},
                'Non-CHD': {"alpha": -26.7, "p": 5.64}}
LOW_RISK_WOMEN = {'CHD': {"alpha": -29.8, "p": 6.36},
                  'Non-CHD': {"alpha": -31.0, "p": 6.62}}
HIGH_RISK_MEN = {'CHD': {"alpha": -21.0, "p": 4.62},
                 'Non-CHD': {"alpha": -25.7, "p": 5.47}}
HIGH_RISK_WOMEN = {'CHD': {"alpha": -28.7, "p": 6.23},
                   'Non-CHD': {"alpha": -30.0, "p": 6.42}}

# coefficients for measurements
COEFF = {'CHD': [0.71, 0.24, 0.018],
         'Non-CHD': [0.63, 0.02, 0.022]}


def _so(age, alpha, p):
    """
    s0(age) = exp{-exp(alpha)*(age)^p}
    """
    return np.exp(-(np.exp(alpha))*(age)**p)


def _baseline_s0(age, coef):
    """
    Calculate the baseline survival probability
    for age and age+10 using the formula:
    s_0(age) = exp{-exp(alpha)*(age-20)^p}
    s_0(age+10) = exp{-exp(alpha)*(age-10)^p}
    """
    return {"s_now": _so(age-20, coef["alpha"], coef["p"]),
            "s_10": _so(age-10, coef["alpha"], coef["p"])}


def _w(chol, sbp, smoking, coef):
    X = np.array([bool(smoking),
                  chol-6,
                  sbp-120])
    return X.dot(coef)


def _survival(baseline_s0, w):
    return {k: v**np.exp(w) for k, v in baseline_s0.items()}


def _risk_10(survival):
    s10_age = survival["s_10"] / survival["s_now"]
    return 1 - s10_age


def _calculate_score(coef_gender, age, chol, sbp, smoking):
    # sum up the two risks
    cvdRisk = 0

    # calculate the two cases, CHD vs non-CHD
    for k, v in coef_gender.items():
        s0 = _baseline_s0(age, v)
        w = _w(chol, sbp, smoking, COEFF[k])
        s = _survival(s0, w)
        cvdRisk += _risk_10(s)
    return cvdRisk


class Score(BaseRisk):
    low_risk = None
    features = ["index_age",
                "female",
                "cur_smoke",
                "sbp", 
                "chol_tot_mmol"]
    feat_key = features

    def __init__(self, low_risk=True):
        self.low_risk = low_risk

    def score(self, row):
        return score(row["female"],
                     row["index_age"],
                     row["chol_tot_mmol"],
                     row["sbp"],
                     row["cur_smoke"],
                     self.low_risk)


def score(female, age, chol_mmol, sbp, smoking, low_risk):
    sc = None
    beta = HIGH_RISK_MEN
    if female and low_risk:
        beta = LOW_RISK_WOMEN
    elif female:
        beta = HIGH_RISK_WOMEN
    elif low_risk:
        beta = LOW_RISK_MEN
    sc = _calculate_score(beta,
                          max(20, age),
                          clean_chol(chol_mmol),
                          clean_bp(sbp),
                          smoking)
    return max(0, min(sc, 1))

