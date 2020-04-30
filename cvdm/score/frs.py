"""
Framingham Risk Score Calculation

Code borrowed from
https://github.com/fonnesbeck/framingham_risk
"""
import numpy as np

from cvdm.score import cox_surv, BaseRisk


NONLAB_WOMEN  = {
    "coef": np.array([2.72107, # log age
                      0.51125, # BMI
                      2.81291, # log SBP (not treated)
                      2.88267, # log SBP (treated)
                      0.61868, # smoking
                      0.77763 # diabetes
                    ]),
    "s0": 0.94833,
    "const": 26.0145
}
NONLAB_MEN  = {
    "coef": np.array([3.11296,
                      0.79277,
                      1.85508,
                      1.92672,
                      0.70953,
                      0.53160
                    ]),
    "s0": 0.88431,
    "const": 23.9388
}
LAB_MEN  = {
    "coef": np.array([3.06117,  # log age
                      1.12370,  # log total cholesterol
                     -0.93263,  # log HDL cholesterol
                      1.93303,  # log SBP (not treated)
                      1.99881,  # log SBP (treated)
                      0.65451,  # smoking
                      0.57367   # diabetes
                    ]),
    "s0": 0.88936,
    "const": 23.9802
}
LAB_WOMEN  = {
    "coef": np.array([2.32888,
                      1.20904,
                     -0.70833,
                      2.76157,
                      2.82263,
                      0.52873,
                      0.69154
                    ]),
    "s0": 0.95012,
    "const": 26.1931
}


class FrsSimple(BaseRisk):
    features = ["female",
                "cur_smoke",
                "dm"]

    def score(self, row):
        return frs_simple(row["female"],
                          row["index_age"],
                          row["bmi"],
                          row["sbp"],
                          row["htn_treat"],
                          row["cur_smoke"],
                          row["dm"])

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["age_log"] = np.log(row["index_age"])
        feat_dict["bmi_log"] = np.log(row["bmi"])
        feat_dict["sbp_nhtn"] = np.log(row["sbp"])*(1-row["htn_treat"])
        feat_dict["sbp_htn"] = np.log(row["sbp"])*(row["htn_treat"])
        return feat_dict


class FrsPrimary(BaseRisk):
    features = ["female",
                "cur_smoke",
                "dm"]

    def score(self, row):
        return frs_primary(row['female'],
                           row["index_age"],
                           row["chol_tot"],
                           row["chol_hdl"],
                           row["sbp"],
                           row["htn_treat"],
                           row['cur_smoke'],
                           row["dm"])

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["age_log"] = np.log(row["index_age"])
        feat_dict["tot_log"] = np.log(row["chol_tot"])
        feat_dict["hdl_log"] = np.log(row["chol_hdl"])
        feat_dict["sbp_nhtn"] = np.log(row["sbp"])*(1-row["htn_treat"])
        feat_dict["sbp_htn"] = np.log(row["sbp"])*(row["htn_treat"])
        return feat_dict
 

def frs_simple(isFemale, age, bmi, sbp, htn, smk, diab):
    """
    10-year risk calculated using the Simple Non-Laboratory 
    Framingham Risk Score (FRS) Calculation.
    
    Parameters
    ----------
    isFemale : boolean
    age : numeric
            Age of subject
    bmi : numeric
            BMI of subject
    sbp : numeric
            Systolic blood pressure of subject
    ht_treat : bool or int
            Treatment for hypertension (True or False)
    smk : bool or int
            Subject is smoker (True or False)
    diab : bool or int
            Subject has diabetes (True or False)
    """
    xFeat = np.array([np.log(age),
                      np.log(bmi),
                      np.log(sbp)*(1-htn),
                      np.log(sbp)*htn,
                      smk,
                      diab])
    genderInfo = NONLAB_MEN
    if isFemale: 
        genderInfo = NONLAB_WOMEN
    return cox_surv(xFeat, genderInfo["coef"],
                    genderInfo["s0"], genderInfo["const"])


def frs_primary(isFemale, age, totChol, hdl, sbp, htn, smk, diab):
    """
    """
    xFeat = np.array([np.log(age),
                      np.log(totChol),
                      np.log(hdl),
                      np.log(sbp)*(1-htn),
                      np.log(sbp)*htn,
                      smk,
                      diab])
    genderInfo = LAB_MEN
    if isFemale: 
        genderInfo = LAB_WOMEN
    return cox_surv(xFeat, genderInfo["coef"],
                    genderInfo["s0"], genderInfo["const"])
