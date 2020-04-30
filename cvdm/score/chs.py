"""
CHS

See: Mukamal, K. J., et al.
"Prediction and classification of cardiovascular disease risk
in older adults with diabetes." Diabetologia 56.2 (2013): 275-283.


"""
import numpy as np

from cvdm.score import BaseRisk


# coefficients for survival
CHS_COEFF = np.log(
                np.array([1.05,  # age in years
                          1.29,  # former smoker
                          1.64,  # current smoker
                          1.15,  # systolic bp / 10 mmHG up to 160
                          1.17,  # total cholestrol per mmmol/l
                          0.79,  # hdl-cholsterol per mmol/l
                          1.43,  # creatine > 110.5 umol/l
                          1.71   # oral hypoglycaemic agent or use
        ]))
MESA_COEFF = np.log(
                np.array([1.03,  # age in years
                          1.24,  # former smoker
                          1.05,  # current smoker
                          1.15,  # systolic bp / 10 mmHG up to 160
                          1.16,  # total cholestrol per mmmol/l
                          0.48,  # hdl-cholsterol per mmol/l
                          2.15,  # creatine > 110.5 umol/l
                          1.89   # oral hypoglycaemic agent or use
        ]))


def chs(age, prevSmoker, curSmoker, sbp, totChol, 
        hdlChol, creatinine, useInsulin, coef="CHS"):
    finalCoef = CHS_COEFF
    if coef == "MESA":
        finalCoef = MESA_COEFF
    xFeat = np.array([age,
                     prevSmoker,
                     curSmoker,
                     min(sbp, 160) / 10,
                     totChol,
                     hdlChol,
                     creatinine > 110.5,
                     useInsulin
        ])
    return xFeat.dot(finalCoef)


class Chs(BaseRisk):
    base_hazard = None
    coef = None
    
    features = ["index_age",
                "prev_smoke",
                "cur_smoke",
                "chol_tot",
                "chol_hdl",
                "insulin"]

    def __init__(self, baseHazard=0.5, coef="CHS"):
        self.base_hazard = baseHazard
        self.coef = coef

    def score(self, row):
        xb = chs(row["index_age"],
                 row["prev_smoke"],
                 row["cur_smoke"],
                 row["sbp"],
                 row["chol_tot"],
                 row["chol_hdl"],
                 row["creat"],
                 row["insulin"],
                 self.coef)
        return 1 - self.base_hazard ** xb
    
