"""
DMCx

Wan, Eric Yuk Fai, et al.
"Development of a cardiovascular diseases risk prediction model
and tools for Chinese patients with type 2 diabetes mellitus:
A population‐based retrospective cohort study." 
Diabetes, Obesity and Metabolism 20.2 (2018): 309-318.
https://www.fmpc.hku.hk/resources/DMCx_RiskEngine.php

"""
import numpy as np

from cvdm.score import cox_surv, BaseRisk


# coefficients for survival
MALE_DCMX = {
    "coef": np.array([   0.0859769,  # age in years
                         0.1869697,  # egfr >= 60 & < 90
                         0.4899165,  # egfr >= 30 & egfr < 60
                         0.9179563,  # egfr < 30
                         0.3523520,  # tc/hdl ratio
                         0.1324850,  # log of urine acr + 1 (mg/mmol)
                         0.8802682,  # smoker
                         0.0104963,  # diabetes duration
                        -0.0247520,  # sbp
                         0.2192236,  # hba1c
                         0.2036549,  # anti-hypertension medications
                        -0.0625381,  # dbp
                        -0.0736360,  # bmi
                         0.2543436,  # insulin use
                         0.0003717,  # dbp^2
                         0.0016522,  # bmi^2
                         0.0001058,  # sbp^2
                         0.0000000,  # hba1c^2
                        -0.0036788,  # age * tc/hdl  ratio
                        -0.0024694,  # age * hba1c
                        -0.0093979,  # age * smoker
                         0.0000000   # anti-glucose medication
        ]),
    "sm": 0.9130526,
    "const": 2.064357
}

FEMALE_DCMX = {
    "coef": np.array([   0.0824519,  # age in years
                         0.1510527,  # egfr >= 60 & < 90
                         0.5176759,  # egfr >= 30 & egfr < 60
                         1.0102600,  # egfr < 30
                         0.3722201,  # tc/hdl ratio
                         0.1222478,  # log of urine acr + 1
                         0.5551439,  # smoker
                         0.0120690,  # diabetes duration
                         0.0043558,  # sbp
                        -0.3121153,  # hba1c
                         0.2888684,  # anti-hypertension medications
                        -0.0807093,  # dbp
                        -0.0614234,  # bmi
                         0.3685551,  # insulin use
                         0.0005025,  # dbp^2
                         0.0014684,  # bmi^2
                         0.0000000,  # sbp^2
                         0.0218833,  # hba1c^2
                        -0.0043098,  # age * tc/hdl  ratio
                         0.0000000,  # age * hba1c
                         0.0000000,  # age * smoker
                         0.1880766   # anti-glucose oral drugs used
        ]),
    "sm": 0.9388678,
    "const": 2.007179
}


def get_dmcx_feat(age, egfr, tchdl, acr, isSmoker, diabDur,
                  sbp, dbp, hba1c, htnMed, bmi, insulin, aGlucose):
    return np.array([age,
                     egfr >= 60 and egfr < 90,
                     egfr >= 30 and egfr < 60,
                     egfr < 30,
                     tchdl,
                     np.log(acr + 1),
                     isSmoker,
                     diabDur,
                     sbp,
                     hba1c,
                     htnMed,
                     dbp,
                     bmi,
                     insulin,
                     dbp**2,
                     bmi**2,
                     sbp**2,
                     hba1c**2,
                     age * tchdl,
                     age * hba1c,
                     age * isSmoker,
                     aGlucose])


def dmcx(age, egfr, tchdl, acr, isSmoker, diabDur, isFemale, 
         sbp, dbp, hba1c, htnMed, bmi, insulin, aGlucose):
    """
    Calculate the risk for cardiovascular disease
    using the coefficients from the DMCX Cohort

    Parameters
    ----------
    age : numeric
            Age of subject
    """
    xFeat = get_dmcx_feat(age, egfr, tchdl, acr, isSmoker, diabDur,
                          sbp, dbp, hba1c, htnMed, bmi, insulin, aGlucose)
    coefInfo = MALE_DCMX
    if isFemale:
        coefInfo = FEMALE_DCMX
    return cox_surv(xFeat, coefInfo["coef"],
                    coefInfo["sm"], coefInfo["const"])



class Dmcx(BaseRisk):
    features = ["index_age",
                "egfr",
                "tchdl",
                "albumin_creat_mgmmol",
                "cur_smoke",
                "diab_dur",
                "female",
                "sbp",
                "dbp",
                "hba1c",
                "htn_treat",
                "bmi",
                "insulin",
                "a_glucose"]
    feat_key = features
    
    def score(self, row):
        return dmcx(row["index_age"],
                    row["egfr"],
                    row["tchdl"],
                    row["albumin_creat_mgmmol"],
                    row["cur_smoke"],
                    row["diab_dur"],
                    row["female"],
                    row["sbp"],
                    row["dbp"],
                    row["hba1c"],
                    row["htn_treat"],
                    row["bmi"],
                    row["insulin"],
                    row["a_glucose"])
