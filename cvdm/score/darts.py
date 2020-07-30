"""
DARTS

See: Donnan PT, Donnelly L, New JP, Morris AD.
Derivation and validation of a prediction score for major
coronary heart disease events in a U.K. type 2 diabetic population.
Diabetes Care. 2006;29:1231-36.

"""
import numpy as np

from cvdm.score import weibull_atf_surv, BaseRisk


# coefficients for survival
BETA = np.array([ -0.287,   # log (duration of diabetes)
                  -0.026,   # age of diagnosis
                  -0.149,   # total cholestrol (mmol/l)
                   0.011,   # former smoker
                  -0.268,   # current smoker
                  -0.308,   # men vs women
                   0.438,   # log of a1c (%)
                  -0.712,   # log of a1c x follow-up (<= 5 vs > 5 years)
                  -0.010,   # sbp
                  -1.292,   # treated htn
                   0.009,   # sbp * treated htn
                   1.241    # height in m 
])
INTERCEPT = 11.262
SIGMA = 0.587


def darts(diabAge, diabDur, cholTot, prevSmoke,
          curSmoke, isMale, hba1c, follow5, sbp,
          htn, height, t):
    if diabDur < 1:
        diabDur = 1 # fix it so that log 1 = 0
    xFeat = np.array([np.log(diabDur),
                     diabAge,
                     cholTot,
                     prevSmoke,
                     curSmoke,
                     isMale,
                     np.log(hba1c),
                     np.log(hba1c)*follow5,
                     sbp,
                     htn,
                     sbp*htn,
                     height])
    s = weibull_atf_surv(xFeat, BETA, INTERCEPT, SIGMA, t)
    return s



class Darts(BaseRisk):
    tYear = None
    features = ["diab_age",
                "chol_tot_mmol",
                "prev_smoke",
                "cur_smoke",
                "male",
                "sbp",
                "htn_treat",
                "height_m"]
    feat_key = features + ["hba1c", "5y_follow"]
    
    
    def __init__(self, tYear=5):
        self.tYear = tYear

    def score(self, row):
        return darts(row["diab_age"],
                     row["diab_dur"],
                     row["chol_tot_mmol"],
                     row["prev_smoke"],
                     row["cur_smoke"],
                     row["male"],
                     row["hba1c"],
                     row["5y_follow"],
                     row["sbp"],
                     row["htn_treat"],
                     row["height_m"],
                     self.tYear)

    def get_features(self, row):
        feat_dict = super().get_features(row)
        diabDur = row["diab_dur"]
        if diabDur < 1:
            diabDur = 1 # fix it so that log 1 = 0
        feat_dict["diab_dur_log"] = np.log(diabDur)
        feat_dict["hba1c_log"] = np.log(row["hba1c"])
        feat_dict["hba1c_log_follow5"] = np.log(row["hba1c"]) * row["5y_follow"]
        feat_dict["sbp_htn"] = row["sbp"] * row["htn_treat"]
        return feat_dict




