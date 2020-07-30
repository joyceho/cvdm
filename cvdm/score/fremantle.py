"""
FREMANTLE

See: Davis WA, Knuiman MW, Davis TM.
An Australian cardiovascular risk equation for type 2 diabetes:
the Fremantle Diabetes Study.
Intern Med J. 2010;40:286-92

"""
import numpy as np

from cvdm.score import cox_surv, BaseRisk


# coefficients for survival
FREMANTLE_SM = 0.904
FREMANTLE_CONST = 7.371
FREMANTLE_COEF = np.array([ 0.080,  # age in years
                            0.335,  # if male
                            0.693,  # prior cvd
                            0.872,  # log of hba1c %
                            0.196,  # log of acr (mg/mmol)
                           -0.608,  # hdl (mmol / L)
                           -0.771,  # southern european
                            1.269   # aboriginal
        ])


def fremantle(age, isMale, cvd, hba1c, acr,
              hdl, seurope, aboriginal):
    """
    Calculate the risk for cardiovascular disease
    using the coefficients from the Fremantle Cohort

    Parameters
    ----------
    age : numeric
            Age of subject
    isMale : boolean or int
            Subject is male (True or False)
    cvd: boolean or int
            Previous history of CVD (True or False)
    hba1c : numeric
            Hba1c (%) of subject
    acr : numeric
            Urinary albumin : creatinine ratio in mg/mmol
    hdl : numeric
            High density lipid cholestrol in mmol/L
    seurope : bool or int
            Subject is Southern European (True or False)
    aboriginal : bool or int
            Subject is Indigenous Australian (True or False)
    """
    xFeat = np.array([age,
                     isMale,
                     cvd,
                     np.log(hba1c),
                     np.log(acr),
                     np.log(hdl),
                     seurope,
                     aboriginal])
    return cox_surv(xFeat, FREMANTLE_COEF,
                    FREMANTLE_SM, FREMANTLE_CONST)


class Fremantle(BaseRisk):
    features = ["index_age",
                "male",
                "cvd_hist",
                "SEuro",
                "Abor"]
    feat_key = features + ["hba1c",
                           "albumin_creat_mgmmol",
                           "chol_hdl_mmol"]

    def score(self, row):
        return fremantle(row["index_age"],
                         row["male"],
                         row["cvd_hist"],
                         row["hba1c"],
                         row["albumin_creat_mgmmol"],
                         row["chol_hdl_mmol"],
                         row["SEuro"],
                         row["Abor"])

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["hba1c_log"] = np.log(row["hba1c"])
        feat_dict["acr_log"] = np.log(row["albumin_creat_mgmmol"])
        feat_dict["hdl_log"] = np.log(row["chol_hdl_mmol"])
        return feat_dict
