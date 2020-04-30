"""
NDR

See: Zethelius, Björn, et al.
"A new model for 5-year risk of cardiovascular disease in 
type 2 diabetes, from the Swedish National Diabetes Register (NDR)." 
Diabetes research and clinical practice 93.2 (2011): 276-284.


"""
import numpy as np

from cvdm.score import cox_surv, BaseRisk


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


def get_ndr_feat(diabAge, diabDur, tchdl, hba1c, sbp, bmi,
                 male, smoker, microalbum, macroalbum,
                 afib, cvd):
    """
    Calculate the survival value
    """
    return np.array([diabAge-53.858,
                     diabDur-7.7360,
                     np.log(tchdl)-1.3948,
                     np.log(hba1c)-1.9736,
                     np.log(sbp)-4.9441,
                     np.log(bmi)-3.3718,
                     male-0.6005,
                     smoker-0.1778,
                     microalbum-0.1604,
                     macroalbum-0.0638,
                     afib-0.0319,
                     cvd-0.1525])


def ndr(diabAge, diabDur, tchdl, hba1c, sbp, bmi,
        male, smoker, microalbum, macroalbum,
        afib, cvd, risk=5):
    if risk not in [4, 5]:
        raise NotImplementedError("Does not support risk that is not 4 or 5")
    baseSurv = S0_4
    if risk == 5:
        baseSurv = S0_5
    xFeat = get_ndr_feat(diabAge, diabDur, tchdl, hba1c,
                         sbp, bmi, male, smoker, microalbum,
                         macroalbum, afib, cvd)
    s = cox_surv(xFeat, BETA, baseSurv)
    return s


NDR_FEAT_MAP = {"smoke_status": 1,
                "hba1c": 1,
                "sbp": 1,
                "history_cvd": 0,
                "afib": 0,
                "diab_age": 0,
                "diab_dur": 0,
                "bmi": 0,
                "tchdl": 0,
                "microalbum": 0,
                "macroalbum": 0}


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

