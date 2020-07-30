"""
DCS

See: https://www.nzssd.org.nz/cvd/

"""
import numpy as np

from cvdm.score import cox_surv, BaseRisk


# coefficients for survival
CVD_INFO = {
    "coef": np.array([ 0.04170,  # age of diagnosis
                      -0.17301,  # gender
                       0.07998,  # previous smoker
                       0.24776,  # current smoker
                       0.05667,  # hba1c
                       0.00489,  # sbp
                       0.20283,  # Maori
                      -0.03110,  # east asian
                       0.08045,  # pacific
                       0.28188,  # indo asian
                       0.24640,  # other
                       0.01628,  # chol ratio
                       0.18900,  # microalbuminuria
                       0.62750,  # macroalbuminuria
                       0.05778,  # duration
                       0.79968,  # hypertension medicine
                      -0.00444   # sbp * hypertension medicine
    ]),
    "s0": 0.817,
    "const": 4.00425997
}

MI_INFO = {
    "coef": np.array([ 0.05274,  # age of diagnosis
                      -0.26065,  # gender
                       0.07915,  # previous smoker
                       0.34834,  # current smoker
                       0.05962,  # hba1c
                       0.00516,  # sbp
                       0.16271,  # Maori
                      -0.15539,  # east asian
                       0.03224,  # pacific
                       0.09004,  # indo asian
                       0.21986,  # other
                       0.03545,  # chol ratio
                       0.17951,  # microalbuminuria
                       0.61812,  # macroalbuminuria
                       0.06931,  # duration
                       0.99161,  # hypertension medicine
                      -0.00592   # sbp * hypertension medicine
    ]),
    "s0": 0.9230,
    "const": 4.8066
}
    

def dcs(diabAge, isFemale, prevSmoke, curSmoke, hba1c, sbp,
        maori, easian, pacific, indoasian, other, tchdl,
        microalbumin, macroalbumin, diabDur, htn, target="CVD"):
    coefInfo = MI_INFO
    if target == "CVD":
        coefInfo = CVD_INFO
    xFeat = np.array([diabAge,
                      isFemale,
                      prevSmoke,
                      curSmoke,
                      hba1c,
                      sbp,
                      maori,
                      easian,
                      pacific,
                      indoasian,
                      other,
                      tchdl,
                      microalbumin,
                      macroalbumin,
                      diabDur,
                      htn,
                      sbp*htn])
    return cox_surv(xFeat, coefInfo["coef"],
                    coefInfo["s0"], coefInfo["const"])


class Dcs(BaseRisk):
    target = None
    features = ["diab_age",
                "female",
                "prev_smoke",
                "hba1c",
                "cur_smoke",
                "sbp",
                "Maori",
                "EAsian",
                "Pacific",
                "IndoAsian",
                "ODcs",
                "tchdl",
                "microalbum",
                "macroalbum",
                "diab_dur",
                "htn_treat"]
    feat_key = features

    def __init__(self, target="CVD"):
        self.target = target

    def score(self, row):
        return dcs(row["diab_age"],
                   row["female"],
                   row["prev_smoke"],
                   row["cur_smoke"],
                   row["hba1c"],
                   row["sbp"],
                   row["Maori"],
                   row["EAsian"],
                   row["Pacific"],
                   row["IndoAsian"],
                   row["ODcs"],
                   row["tchdl"],
                   row["microalbum"],
                   row["macroalbum"],
                   row["diab_dur"],
                   row["htn_treat"],
                   target=self.target)

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["sbp_htn"] = feat_dict["sbp"]*feat_dict["htn_treat"]
        return feat_dict
