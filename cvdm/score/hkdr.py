"""
HKDR

See: Yang X, So WY, Kong AP, Ma RC, Ko GT, Ho CS et al.
Development and validation of a total
coronary heart disease risk score in type 2 diabetes mellitus.
Am J Cardiol. 2008;101:596-601.

"""
import numpy as np

from cvdm.score import cox_surv, BaseRisk


# coefficients for survival
HKDR_CHD = {
    "coef": np.array([ 0.0267,  # age in years
                      -0.3536,  # if female
                       0.4373,  # current smoker
                       0.0403,  # duration of diabetes
                      -0.4808,  # log_10 of efgr (ml/min/1.73 m^2)
                       0.1232,  # log_10 of (1+spot ACR) (mg/mmol)
                       0.2644   # non-hdl cholestrol (mmol/L)
        ]),
    "sm": 0.9616,
    "const": 0.7082,
    "shrink": 0.9440
}


HKDR_HF = {
    "coef": np.array([ 0.0709,  # age in years
                       0.0627,  # bmi
                       0.1363,  # hba1c (%)
                       0.9915,  # log_10 of (1+spot ACR) (mg/mmol
                      -0.3606,  # blood HB (g/dl)
                       0.8161   # chd during followup
        ]),
    "male_sm": 0.9888,
    "female_sm": 0.9809,
    "const": 2.3961,
    "shrink": 0.9744
}

HKDR_STROKE = {
    "coef": np.array([ 0.0634,  # age in years
                       0.0897,  # hba1c (%)
                       0.5314,  # log_10 ACR
                       0.5636   # hist_chd
                       ]),
    "const": 4.5674,
    "sm": 0.9707
}


def hkdr_chd(age, isFemale, curSmoke, diabDur,
             egfr, acr, nonHdlChol):
    """
    Calculate the risk for coronary heart disease
    using the coefficients from the HKDR CHD Cohort

    Parameters
    ----------
    age : numeric
            Age of subject
    isFemale : boolean or int
            Subject is female (True or False)
    curSmoke: boolean or int
            Previous history of CVD (True or False)
    diabDur : numeric
            Nubmer of years of diabetes
    egfr : numeric
            Estimated Glomerular Filteration Rate
    acr : numeric
            Urinary albumin : creatinine ratio in mg/mmol
    nonHDL : numeric
            Non-HDL cholesterol (mmol/L)
    """
    xFeat = np.array([age,
                      isFemale,
                      curSmoke,
                      diabDur,
                      np.log10(egfr),
                      np.log10(1+acr),
                      nonHdlChol])
    return cox_surv(xFeat,
                    HKDR_CHD["coef"],
                    HKDR_CHD["sm"],
                    HKDR_CHD["const"],
                    HKDR_CHD["shrink"])


class HkdrCHD(BaseRisk):
    features = ["female",
                "index_age",
                "diab_dur",
                "cur_smoke",
                "nonhdl_mmol"]
    feat_key = features + ["egfr",
                           "albumin_creat_mgmmol"]

    def score(self, row):
        return hkdr_chd(row["index_age"],
                        row["female"],
                        row["cur_smoke"],
                        row["diab_dur"],
                        row["egfr"],
                        row["albumin_creat_mgmmol"],
                        row["nonhdl_mmol"])

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["egfr_log"] = np.log10(row["egfr"])
        feat_dict["acr_log"] = np.log10(1+row["albumin_creat_mgmmol"])
        return feat_dict


def hkdr_hf(isFemale, age, bmi, hba1c, acr, hb, chdHist):
    """
    Calculate the risk for heart failure
    using the coefficients from the HKDR Cohort

    Parameters
    ----------
    age : numeric
            Age of subject
    bmi : numeric
            BMI of the subject (in kg/m^2)
    hba1c: numeric
            HBA1C (%)
    acr : numeric
            Urinary albumin : creatinine ratio in mg/mmol
    hb : numeric
            Blood Hemoglobin (g/dl)
    chdHist : boolean or int
            Subject had CHD (true or False)
    """
    baseSurv = HKDR_HF["male_sm"]
    if isFemale:
        baseSurv = HKDR_HF["female_sm"]
    xFeat = np.array([age,
                     bmi,
                     hba1c,
                     np.log10(1+acr),
                     hb,
                     chdHist])
    return cox_surv(xFeat,
                    HKDR_HF["coef"],
                    baseSurv,
                    HKDR_HF["const"],
                    HKDR_HF["shrink"])


class HkdrHF(BaseRisk):
    features = ["female",
                "index_age",
                "bmi",
                "hba1c",
                "hb",
                "chd"]
    feat_key = features + ["albumin_creat_mgmmol"]

    def score(self, row):
        return hkdr_hf(row["female"],
                       row["index_age"],
                       row["bmi"],
                       row["hba1c"],
                       row["albumin_creat_mgmmol"],
                       row["hb"],
                       row["chd"])


    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["acr_log"] = np.log10(1+row["albumin_creat_mgmmol"])
        return feat_dict



def hkdr_stroke(age, hba1c, acr, chd):
    xFeat = np.array([age,
                      hba1c,
                      np.log10(acr),
                      chd])
    return cox_surv(xFeat,
                    HKDR_STROKE["coef"],
                    HKDR_STROKE["sm"],
                    HKDR_STROKE["const"])


class HkdrStroke(BaseRisk):
    features = ["index_age",
                "hba1c",
                "chd"]
    feat_key = features + ["albumin_creat_mgmmol"]
    
    def score(self, row):
        return hkdr_stroke(row["index_age"],
                           row["hba1c"],
                           row["albumin_creat_mgmmol"],
                           row["chd"])

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["acr_log"] = np.log10(1+row["albumin_creat_mgmmol"])
        return feat_dict
