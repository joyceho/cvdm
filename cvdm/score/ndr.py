"""
NDR

See: Zethelius, Björn, et al.
"A new model for 5-year risk of cardiovascular disease in 
type 2 diabetes, from the Swedish National Diabetes Register (NDR)." 
Diabetes research and clinical practice 93.2 (2011): 276-284.


"""
import numpy as np

from cvdm.score import cox_surv, BaseRisk
from cvdm.score import clean_diab_dur, clean_hba1c, clean_bp, clean_bmi, clean_tchdl


# coefficients for survival
BETA = np.array([ 0.0498, # age at diagnosis of diabetes
                  0.0651, # duration of diabetes
                  0.5737, # log of tc:hdl
                  0.6929, # log of hba1c
                  0.7055, # log of sbp
                  0.4105, # log of bmi (kg / m)
                  0.3438, # male
                  0.2998, # smoker
                  0.2414, # microalbuminuria
                  0.4252, # macroalbuminuria
                  0.4034, # afib
                  0.6838  # previous cvd
])
S0_5 = 0.90237
S0_4 = 0.92347


def ndr(diab_age, diab_dur, tchdl, hba1c, sbp, bmi,
        male, smoker, microalbum, macroalbum,
        afib, cvd, risk=5):
    if risk not in [4, 5]:
        raise NotImplementedError("Does not support risk that is not 4 or 5")
    baseSurv = S0_4
    if risk == 5:
        baseSurv = S0_5
    xFeat = np.array([diab_age-53.858,
                      clean_diab_dur(diab_dur)-7.7360,
                      np.log(clean_tchdl(tchdl))-1.3948,
                      np.log(clean_hba1c(hba1c))-1.9736,
                      np.log(clean_bp(sbp))-4.9441,
                      np.log(clean_bmi(bmi))-3.3718,
                      male-0.6005,
                      smoker-0.1778,
                      microalbum-0.1604,
                      macroalbum-0.0638,
                      afib-0.0319,
                      cvd-0.1525])
    s = cox_surv(xFeat, BETA, baseSurv)
    return s


class Ndr(BaseRisk):
    risk = None
    features = ["diab_age",
                "diab_dur",
                "male",
                "cur_smoke",
                "microalbum",
                "macroalbum",
                "afib",
                "cvd_hist"]
    feat_key = features + ["tchdl",
                           "hba1c",
                           "sbp",
                           "bmi"]


    def __init__(self, risk = 5):
        self.risk = risk

    def score(self, row):
        return ndr(row["diab_age"], 
                   row["diab_dur"],
                   row["tchdl"],
                   row["hba1c"],
                   row["sbp"],
                   row["bmi"],
                   row["male"],
                   row["cur_smoke"],
                   row["microalbum"],
                   row["macroalbum"],
                   row["afib"],
                   row["cvd_hist"],
                   self.risk)

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["tchdl_log"] = np.log(row["tchdl"])
        feat_dict["hba1c_log"] = np.log(row["hba1c"])
        feat_dict["sbp"] = np.log(row["sbp"])
        feat_dict["bmi"] = np.log(row["bmi"])
        return feat_dict

