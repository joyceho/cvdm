"""
Pooled Cohort Equation

https://clincalc.com/cardiology/ascvd/pooledcohort.aspx
"""
import numpy as np

from cvdm.score import cox_surv, BaseRisk
from cvdm.score import clean_age, clean_tot_chol, clean_hdl, clean_bp


WHITE_FEMALE  = {
    "coef": np.array([-29.799,   # log of age
                        4.884,   # log of age**2
                       13.540,   # log(total chol)
                       -3.114,   # log(total chol) * log(age)
                      -13.578,   # log(hdl)
                        3.149,   # log(hdl)* log(age)
                        1.957,   # log(sbp) * (1-htn)
                        0.000,   # log(age) * log(sbp) * (1-htn)
                        2.019,   # log(sbp) * (htn)
                        0.000,   # log(age) * log(sbp) * (htn)
                        7.574,   # smoker
                       -1.665,   # smoker * log(age)
                        0.661    # diabetes
                    ]),
    "s5": 0.98898,
    "s10": 0.9665,
    "const": -29.18
}

BLACK_FEMALE  = {
    "coef": np.array([ 17.114,   # log of age
                        0.000,   # log of age**2
                        0.940,   # log(total chol)
                        0.000,   # log(total chol) * log(age)
                      -18.920,   # log(hdl)
                        4.475,   # log(hdl)* log(age)
                       27.820,   # log(sbp) * (1- htn)
                       -6.087,   # log(age) * log(sbp) * (1-htn)
                       29.291,   # log(sbp) * (htn)
                       -6.432,   # log(age) * log(sbp) * (htn)
                        0.691,   # smoker
                        0.000,   # smoker * log(age)
                        0.874    # diabetes
                    ]),
    "s5": 0.98194,
    "s10": 0.9533,
    "const": 86.61
}


WHITE_MALE  = {
    "coef": np.array([ 12.344,   # log of age
                        0.000,   # log of age**2
                       11.853,   # log(total chol)
                       -2.664,   # log(total chol) * log(age)
                       -7.990,   # log(hdl)
                        1.769,   # log(hdl)* log(age)
                        1.764,   # log(sbp) * (1 - htn)
                        0.000,   # log(age) * log(sbp) * (1 - htn)
                        1.797,   # log(sbp) * htn
                        0.000,   # log(age) * log(sbp) * htn
                        7.837,   # smoker
                        1.795,   # smoker * log(age)
                        0.658    # diabetes
                    ]),
    "s5": 0.96254,
    "s10": 0.9144,
    "const": 61.18
}

BLACK_MALE  = {
    "coef": np.array([  2.469,   # log of age
                        0.000,   # log of age**2
                        0.302,   # log(total chol)
                        0.000,   # log(total chol) * log(age)
                       -0.307,   # log(hdl)
                        0.000,   # log(hdl)* log(age)
                        1.809,   # log(sbp) * (1-htn)
                        0.000,   # log(age) * log(sbp)* (1-htn)
                        1.916,   # log(sbp) * (htn)
                        0.000,   # log(age) * log(sbp)* (htn)
                        0.549,   # smoker
                        0.000,   # smoker * log(age)
                        0.645    # diabetes
                    ]),
    "s5": 0.95726,
    "s10": 0.8954,
    "const": 19.54
}


def pce(female, ac, age, tot_chol, hdl,
        sbp, smoker, htn, diab, risk=5):
    if risk not in [5, 10]:
        raise NotImplementedError("Does not support risk that is not 5 or 10")
    baseSurv = "s10"
    if risk == 5:
        baseSurv = "s5"
    # figure out what the betas are
    cohortInfo = WHITE_MALE
    if female and ac:
        cohortInfo = BLACK_FEMALE
    elif female:
        cohortInfo = WHITE_FEMALE
    elif ac:
        cohortInfo = BLACK_MALE
    age = clean_age(age)
    tot_chol = clean_tot_chol(tot_chol)
    hdl = clean_hdl(hdl)
    sbp = clean_bp(sbp)
    xFeat = np.array([np.log(age),
                      np.log(age)**2,
                      np.log(tot_chol),
                      np.log(tot_chol)*np.log(age),
                      np.log(hdl),
                      np.log(hdl)*np.log(age),
                      np.log(sbp)*(1-htn),
                      np.log(age)*np.log(sbp)*(1-htn),
                      np.log(sbp)*htn,
                      np.log(age)*np.log(sbp)*htn,
                      smoker,
                      smoker*np.log(age), 
                      diab])
    s = cox_surv(xFeat, cohortInfo["coef"],
                 cohortInfo[baseSurv],
                 cohortInfo["const"])
    return s


class Pce(BaseRisk):
    risk = None
    features = ["female",
                "AC",
                "cur_smoke",
                "dm"]
    feat_key = features + ["index_age",
                           "chol_tot", 
                           "chol_hdl", 
                           "sbp",
                           "htn_treat"]

    def __init__(self, risk = 5):
        self.risk = risk

    def score(self, row):
        return pce(row["female"],
                   row["AC"],
                   row["index_age"],
                   row["chol_tot"],
                   row["chol_hdl"],
                   row["sbp"],
                   row["cur_smoke"],
                   row["htn_treat"],
                   row["dm"],
                   self.risk)

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["age_log"] = np.log(row["index_age"])
        feat_dict["age_log2"] = np.log(row["index_age"])**2
        feat_dict["tot_log2"] = np.log(row["chol_tot"])
        feat_dict["tot_age"] = np.log(row["chol_tot"])*feat_dict["age_log"]
        feat_dict["hdl_log2"] = np.log(row["chol_hdl"])
        feat_dict["hdl_age"] = np.log(row["chol_hdl"])*feat_dict["age_log"]
        feat_dict["sbp_nhtn"] = np.log(row["sbp"])*(1-row["htn_treat"])
        feat_dict["sbp_age_nhtn"] = feat_dict["age_log"]*np.log(row["sbp"])*(1-row["htn_treat"])
        feat_dict["sbp_htn"] = np.log(row["sbp"])*row["htn_treat"]
        feat_dict["sbp_age_htn"] = feat_dict["age_log"]*np.log(row["sbp"])*row["htn_treat"]
        feat_dict["smoke_age"] = feat_dict["age_log"]*feat_dict["cur_smoke"]
        return feat_dict
