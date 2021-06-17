"""
DIAL

See Berkelmans, Gijs FN, et al.
"Prediction of individual life-years gained without
cardiovascular events from lipid, blood pressure, glucose,
and aspirin treatment based on data of more than 
500 000 patients with Type 2 diabetes mellitus."
European heart journal 40.34 (2019): 2899-2906.

https://u-prevent.com/calculators/results/dialModel

"""
import numpy as np

from cvdm.score import cox_surv, BaseRisk
from cvdm.score import clean_age, clean_bmi, clean_bp, clean_nonhdl
from cvdm.score import clean_hba1c, clean_egfr, clean_diab_dur


# age-specific baseline survivals
AGE_S0 = {
    30: 0.99828,
    31: 1.00000,
    32: 1.00000,
    33: 0.99883,
    34: 1.00000,
    35: 0.99910,
    36: 0.99857,
    37: 0.99825,
    38: 1.00000,
    39: 1.00000,
    40: 0.99929,
    41: 0.99815,
    42: 0.99922,
    43: 0.99824,
    44: 0.99781,
    45: 0.99770,
    46: 0.99857,
    47: 0.99807,
    48: 0.99757,
    49: 0.99696,
    50: 0.99793,
    51: 0.99722,
    52: 0.99692,
    53: 0.99684,
    54: 0.99626,
    55: 0.99621,
    56: 0.99647,
    57: 0.99702,
    58: 0.99659,
    59: 0.99665,
    60: 0.99649,
    61: 0.99659,
    62: 0.99676,
    63: 0.99670,
    64: 0.99722,
    65: 0.99700,
    66: 0.99712,
    67: 0.99738,
    68: 0.99718,
    69: 0.99726,
    70: 0.99741,
    71: 0.99789,
    72: 0.99782,
    73: 0.99772,
    74: 0.99784,
    75: 0.99792,
    76: 0.99796,
    77: 0.99800,
    78: 0.99797,
    79: 0.99811,
    80: 0.99828,
    81: 0.99822,
    82: 0.99834,
    83: 0.99832,
    84: 0.99845,
    85: 0.99852,
    86: 0.99863,
    87: 0.99875,
    88: 0.99873,
    89: 0.99894,
    90: 0.99907,
    91: 0.99913,
    92: 0.99923,
    93: 0.99943,
    94: 0.99955
}


# coefficients for survival
DIAL_COEF = np.array([-2.432709,     # male
                       0.035983,     # age (if male)
                      -0.08603257,   # BMI - 30
                       0.001155281,  # squared BMI - 30^2
                      -0.6910912,    # smoking
                       0.01127745,   # age (if smoking)
                      -0.02365684,   # sbp - 140
                       0.00009386,   # squared sbp - 140^2
                       0.2632915,    # nonHDL - 3.8
                      -0.02153226,   # squared nonHDL - 3.8^2
                       0.02274024,   # hba1c-50
                      -0.0001292752, # squared hba1c - 50^2
                      -0.01172895,   # egfr-80
                      -0.00002497421,# squared egfr - 80^2
                       0.1654953,    # microalbuminaria
                       0.2061535,    # macroalbuminaria
                       0.01650379,   # diabetes duration
                      -0.4734714,    # history of cvd
                       0.04268836,   # age (if history of cvd)
                      -0.8525590,    # insulin treatment
                       0.01344922,   # age (if insulin treatment)
                       1,            # LN(Hazard Ratio of intended treatment)§,
                       1.763233      # (if high risk county)  
    ])
    

def dial(is_male, age, bmi, cur_smoke,
         sbp, non_hdl, hba1c, egfr,
         microalbumin, macroalbumin,
         diab_dur, cvd_hist, insulin,
         hz_treat=0, high_risk_county=False):
    # fix the age to be within 18-94
    age = min(clean_age(age), 94)
    diab_dur = int(round(clean_diab_dur(diab_dur)))
    bmi = clean_bmi(bmi)
    sbp = clean_bp(sbp)
    non_hdl = clean_nonhdl(non_hdl, meas="mmol")
    egfr = clean_egfr(egfr)
    hba1c = clean_hba1c(hba1c, meas="mmol")
    xFeat = np.array([is_male,
                      age*is_male,
                      bmi-30,
                      bmi**2 - 30**2,
                      cur_smoke,
                      age*cur_smoke,
                      sbp-140,
                      sbp**2 - 140**2,
                      non_hdl - 3.8,
                      non_hdl**2 - 3.8**2,
                      hba1c-50,
                      hba1c**2 - 50**2,
                      egfr - 80,
                      egfr**2 - 80**2,
                      microalbumin,
                      macroalbumin,
                      diab_dur,
                      cvd_hist,
                      age*cvd_hist,
                      insulin,
                      age * insulin,
                      hz_treat,
                      high_risk_county])
    # look-up the age-specific value
    s0 = AGE_S0[age]
    return cox_surv(xFeat, DIAL_COEF, s0)


class Dial(BaseRisk):
    features = ["male",
                "cur_smoke",
                "microalbum",
                "macroalbum",
                "diab_dur",
                "cvd_hist",
                "insulin"]
    feat_key = features + ["index_age",
                           "bmi", "sbp",
                           "nonhdl_mmol",
                           "hba1c_mmol",
                           "egfr"]

    def score(self, row):
        return dial(row["male"],
                    row["index_age"],
                    row["bmi"],
                    row["cur_smoke"],
                    row["sbp"],
                    row["nonhdl_mmol"],
                    row["hba1c_mmol"],
                    row["egfr"],
                    row["microalbum"],
                    row["macroalbum"],
                    row["diab_dur"],
                    row["cvd_hist"],
                    row["insulin"])

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["age_male"] = row["index_age"]*row["male"]
        feat_dict["bmi_30"] = row["bmi"] - 30
        feat_dict["bmi_2_30"] = row["bmi"]**2 - 30**2
        feat_dict["age_smoke"] = row["index_age"]*row["cur_smoke"]
        feat_dict["sbp_140"] = row["sbp"]-140
        feat_dict["sbp_2_140"] = row["sbp"]**2 - 140**2
        feat_dict["nonhdl_38"] = row["nonhdl_mmol"] - 3.8
        feat_dict["nonhdl_2_38"] = row["nonhdl_mmol"]**2 - 3.8**2
        feat_dict["hba1c_50"] = row["hba1c_mmol"] - 50
        feat_dict["hba1c_2_50"] = row["hba1c_mmol"]**2 - 50**2
        feat_dict["egfr_80"] = row["egfr"] - 80
        feat_dict["egfr_2_80"] = row["egfr"]**2 - 80**2
        feat_dict["age_cvd"] = row["index_age"] * row["cvd_hist"]
        feat_dict["age_insulin"] = row["index_age"] * row["insulin"]
        return feat_dict
