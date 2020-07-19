"""
ADVANCE

See: Kengne, Andre Pascal, et al. "Contemporary model for cardiovascular risk prediction in people with type 2 diabetes."
European Journal of Cardiovascular Prevention & Rehabilitation 18.3 (2011): 393-398.


"""
import numpy as np

from cvdm.score import cox_surv, BaseRisk

# coefficients for survival
BETA = np.array([ 0.06187, # age at diagnosis of diabetes
                  -0.4736, # risk for females
                  0.08263, # duration of diabetes
                  0.00665, # pulse pressure (mmHg)
                  0.38248, # retinopathy
                  0.60160, # atrial fibrillation
                  0.09945, # unit increase in hba1c (%)
                  0.19341, # log of albumin/creatine ratio (mg/g)
                  0.12619, # non-hdl cholestrol (mmol/l)
                  0.24219  # treated hypertension
])
S_0 = 0.951044
CONST = 6.52910152


def advance(diabAge, isFemale, diabDur,
            pp, retin, afib,
            hba1c, acr, nonhdl, htn):
    xFeat = np.array([diabAge, isFemale, diabDur,
                      pp, retin, afib, hba1c,
                      np.log(acr), nonhdl, htn])
    s = cox_surv(xFeat, BETA, S_0, CONST)
    return s


class Advance(BaseRisk):
    features = ["diab_age",
                "female",
                "diab_dur",
                "pp",
                "retinopathy",
                "afib",
                "hba1c",
                "albumin_creat",
                "nonhdl_mmoll",
                "htn_treat"]

    def score(self, row):
        return advance(row["diab_age"],
                       row["female"],
                       row["diab_dur"],
                       row["pp"],
                       row["retinopathy"], 
                       row["afib"],
                       row["hba1c"],
                       row["albumin_creat"],
                       row["nonhdl_mmoll"],
                       row["htn_treat"])

    def get_features(self, row):
        """
        Get the features associated with this score
        """
        feat_dict = super().get_features(row)
        # log transform albumin creat
        feat_dict["albumin_creat"] = np.log(feat_dict["albumin_creat"])
        return feat_dict
