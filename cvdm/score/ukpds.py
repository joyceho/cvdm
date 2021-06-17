"""
UKPDS

See:

"""
import numpy as np

from cvdm.score import BaseRisk
from cvdm.score import clean_age, clean_hba1c, clean_bp, clean_tchdl


# coefficients for survival
BETA = np.array([ 1.059, # age at diagnosis of diabetes
                  0.525, # risk for females
                  0.390, # Afro-Carribean ethnicity
                  1.350, # smoking
                  1.183, # HBA1c
                  1.088, # 10mmHg increase in systolic blood pressure
                  3.845  # unit increase in log of lipid ratio
])
Q_0 = 0.0112 # intercept
D = 1.078    # risk ratio for each year increase in duration of diagnosed diabetes


def ukpds(ageDiab, age, female, ac, smoking, hba1c, sbp, tchdl, tYear=10):
    """
    Calculate the number of years to forecast the risk.
    """
    xFeat = np.array([clean_age(age)-55,
                      female,
                      ac,
                      bool(smoking),
                      clean_hba1c(hba1c)-6.72,
                      (clean_bp(sbp) - 135.7)/10,
                      np.log(clean_tchdl(tchdl))-1.59])
    q = Q_0 * np.prod(np.power(BETA, xFeat))
    uscore = 1 - np.exp(-q * D**(age-ageDiab)* (1-D**tYear)/ (1 - D))
    return max(uscore, 0.0)


class Ukpds(BaseRisk):
    tYear = None
    features = ["diab_age",
                "index_age",
                "female",
                "AC",
                "cur_smoke",
                "hba1c",
                "sbp"]
    feat_key = features + ["tchdl"]

    def __init__(self, tYear=10):
        self.tYear = tYear

    def score(self, row):
        return ukpds(row["diab_age"],
                     row["index_age"],
                     row["female"],
                     row["AC"],
                     row["cur_smoke"],
                     row["hba1c"],
                     row["sbp"],
                     row["tchdl"],
                     tYear=self.tYear)

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["tchdl_log"] = np.log(row["tchdl"])
        return feat_dict
