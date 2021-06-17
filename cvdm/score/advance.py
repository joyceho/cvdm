"""
ADVANCE

See: Kengne, Andre Pascal, et al. "Contemporary model for cardiovascular risk prediction in people with type 2 diabetes."
European Journal of Cardiovascular Prevention & Rehabilitation 18.3 (2011): 393-398.


"""
import numpy as np

from cvdm.score import cox_surv, BaseRisk
from cvdm.score import clean_diab_dur, clean_pp, clean_hba1c, clean_acr, clean_nonhdl

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


def advance(diab_age, female, diab_dur,
            pp, retin, afib,
            hba1c, acr, non_hdl, htn_treat):
    # add ability to ensure specific values are not negative
    xFeat = np.array([diab_age,
                      female,
                      clean_diab_dur(diab_dur),
                      clean_pp(pp),
                      retin,
                      afib,
                      clean_hba1c(hba1c),
                      np.log(clean_acr(acr)),
                      clean_nonhdl(non_hdl),
                      htn_treat])
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
                "nonhdl_mmol",
                "htn_treat"]
    # set them to be the same
    feat_key = features

    def score(self, row):
        return advance(row["diab_age"],
                       row["female"],
                       row["diab_dur"],
                       row["pp"],
                       row["retinopathy"], 
                       row["afib"],
                       row["hba1c"],
                       row["albumin_creat"],
                       row["nonhdl_mmol"],
                       row["htn_treat"])

    def get_features(self, row):
        """
        Get the features associated with this score
        """
        feat_dict = super().get_features(row)
        # log transform albumin creat
        feat_dict["albumin_creat"] = np.log(feat_dict["albumin_creat"])
        return feat_dict
