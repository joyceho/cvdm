"""
UKPDS Outcome Model 2

See: Hayes, A. J., et al.
"UKPDS outcomes model 2: a new version of a model to
simulate lifetime health outcomes of patients with 
type 2 diabetes mellitus using data from the 
30 year United Kingdom Prospective Diabetes Study: UKPDS 82." 
Diabetologia 56.9 (2013): 1925-1933.
"""
import numpy as np

from cvdm.score import weibull_surv, BaseRisk


CHF_PARAMS = {"rho": 1.514,
              "lambda": -12.332,
              "beta": np.array([ 0.068, # age at diagnosis of diabetes
                                 1.562, # AFIB (1 for atrial fibrillation; 0 otherwise.Defined from Minnesota codes 831 and833)
                                 0.072, # BMI (m/kg^2)
                                -0.220, # eGFR < 60
                                 0.012, # LDL (*10) in mmol/l
                                 0.771, # MMALB (1 for urine albumin  ≥50mg/l; 0 otherwise)
                                 0.479, # PVD (1 for peripheral vascular disease; 0 otherwise)
                                 0.658, # AMP HIST (1 for history of amputation; 0 otherwise)
                                 0.654  # ULCER HISTORY
                               ])
}

STROKE_PARAMS = {"rho": 1.466,
                 "lambda": -13.053,
                 "beta": np.array([ 0.066, # age at diagnosis of diabetes
                                   -0.420, # female
                                    1.476, # AFIB
                                   -0.190, # eGFR < 60
                                    0.092, # hbA1c
                                    0.016, # LDL (*10) in mmol/l
                                    0.420, # MMALB
                                    0.170, # SBP
                                    0.331, # smoker
                                    0.040, # WBC
                                    1.090, # AMP_HIST
                                    0.481  # CHD_HIST
                                    ])
}


MI_FEMALE_PARAMS = {"rho": 1.376,
                    "lambda":-8.708,
                    "beta": np.array([-1.684, # afro
                                       0.041, # age at diagnosis of diabetes
                                      -0.280, # eGFR < 60
                                       0.078, # hba1c
                                       0.035, # ldl > 35
                                       0.277, # MMALB
                                       0.469, # pvd
                                       0.056, # sbp
                                       0.344, # smoker
                                       0.070, # WBC
                                       0.853, # chf history
                                       0.876  # CHD history
                    ])
}


MI_MALE_PARAMS = {"rho": 1,
                  "lambda":-8.791,
                  "beta": np.array([-0.830, # afro
                                     0.045, # age at diagnosis of diabetes
                                     0.279, # indian
                                     0.108, # hba1c
                                    -0.049, # hdl
                                     0.023, # ldl 
                                     0.203, # MMALB
                                     0.340, # pvd
                                     0.046, # sbp
                                     0.277, # smoker
                                     0.026, # WBC
                                     0.743, # amp history
                                     0.814, # chf history
                                     0.846, # CHD history
                                     0.448  # stroke history
                    ])
}


def ukpdsom2_chf(diabDur, diabAge, afib, bmi, egfr, ldl,
                 mmalb, pvd, ampHist, ulcHist, tYear=1):
    """
    Calculate the number of years to forecast the risk.
    """
    xFeat = np.array([diabAge,
                      afib,
                      bmi,
                      egfr/10 if egfr <60 else 0,
                      ldl*10,
                      mmalb >= 50,
                      pvd,
                      ampHist,
                      ulcHist])
    return weibull_surv(xFeat, CHF_PARAMS["beta"],
                        CHF_PARAMS["lambda"],
                        diabDur, diabDur+tYear,
                        CHF_PARAMS["rho"])


class UkpdsOM2CHF(BaseRisk):
    tYear = None
    features = ["diab_dur",
                "diab_age",
                "afib",
                "bmi",
                "pvd",
                "amp_hist",
                "ulcer_hist"]
    

    def __init__(self, tYear=10):
        self.tYear = tYear

    def score(self, row):
        return ukpdsom2_chf(row["diab_dur"],
                            row["diab_age"],
                            row["afib"],
                            row["bmi"],
                            row["egfr"],
                            row["chol_ldl_mmol"],
                            row["albumin_urine"],
                            row["pvd"],
                            row["amp_hist"],
                            row["ulcer_hist"],
                            tYear=self.tYear)

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["egfr_gt_60"] = row["egfr"]/10 if row["egfr"]<60 else 0
        feat_dict["ldl_mmol_10"] = row["chol_ldl_mmol"]*10
        feat_dict["mmalb"] = row["albumin_urine"] >= 50
        return feat_dict


def ukpdsom2_stroke(diab_dur, diab_age, female, afib,
                    egfr, hba1c, ldl, mmalb,
                    sbp, cur_smoke, wbc, amp_hist,
                    chd_hist, tYear=1):
    """
    Calculate the number of years to forecast the risk.
    """
    xFeat = np.array([diab_age,
                      female,
                      afib,
                      egfr/10 if egfr <60 else 0,
                      hba1c,
                      ldl*10,
                      mmalb >= 50,
                      sbp/10,
                      cur_smoke,
                      wbc,
                      amp_hist,
                      chd_hist])
    return weibull_surv(xFeat, STROKE_PARAMS["beta"],
                        STROKE_PARAMS["lambda"],
                        diab_dur, diab_dur+tYear,
                        STROKE_PARAMS["rho"])


class UkpdsOM2Stroke(BaseRisk):
    tYear = None
    features = ["diab_dur",
                "diab_age",
                "female",
                "afib",
                "hba1c",
                "cur_smoke",
                "wbc",
                "amp_hist",
                "chd"]

    def __init__(self, tYear=10):
        self.tYear = tYear

    def score(self, row):
        return ukpdsom2_stroke(row["diab_dur"],
                               row["diab_age"],
                               row["female"],
                               row["afib"],
                               row["egfr"],
                               row["hba1c"],
                               row["chol_ldl_mmol"],
                               row["albumin_urine"],
                               row["sbp"],
                               row["cur_smoke"],
                               row["wbc"],
                               row["amp_hist"],
                               row["chd"],
                               tYear=self.tYear)

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["egfr_gt_60"] = row["egfr"]/10 if row["egfr"]<60 else 0
        feat_dict["ldl_mmol_10"] = row["chol_ldl_mmol"]*10
        feat_dict["mmalb"] = row["albumin_urine"] >= 50
        feat_dict["sbp_10"] = row["sbp"] / 10
        return feat_dict



def ukpdsom2_mi_male(ac, diab_dur, diab_age, easian, 
                     hba1c, hdl, ldl, mmalb,
                     pvd, sbp, cur_smoke, wbc,
                     amp_hist, chf_hist, chd_hist,
                     stroke_hist, tYear=1):
    """
    Calculate the number of years to forecast the risk.
    """
    xFeat = np.array([ac,
                      diab_age,
                      easian,
                      hba1c,
                      hdl*10,
                      ldl*10,
                      mmalb >= 50,
                      pvd,
                      sbp/10,
                      cur_smoke,
                      wbc,
                      amp_hist,
                      chf_hist,
                      chd_hist,
                      stroke_hist])
    return weibull_surv(xFeat, MI_MALE_PARAMS["beta"],
                        MI_MALE_PARAMS["lambda"],
                        diab_dur, diab_dur+tYear,
                        MI_MALE_PARAMS["rho"])


def ukpdsom2_mi_female(ac, diab_dur, diab_age, egfr,
                       hba1c, ldl, mmalb,
                       pvd, sbp, cur_smoke, wbc,
                       chf_hist, chd_hist, tYear=1):
    """
    Calculate the number of years to forecast the risk.
    """
    xFeat = np.array([ac,
                      diab_age,
                      egfr/10 if egfr <60 else 0,
                      hba1c,
                      ldl*10 if ldl > 35 else 0,
                      mmalb >= 50,
                      pvd,
                      sbp/10,
                      cur_smoke,
                      wbc,
                      chf_hist,
                      chd_hist])
    return weibull_surv(xFeat, MI_FEMALE_PARAMS["beta"],
                        MI_FEMALE_PARAMS["lambda"],
                        diab_dur, diab_dur+tYear,
                        MI_MALE_PARAMS["rho"])


class UkpdsOM2MI(BaseRisk):
    tYear = None
    features = ["AC",
                "EAsian",
                "diab_dur",
                "diab_age",
                "hba1c",
                "cur_smoke",
                "pvd",
                "wbc",
                "amp_hist",
                "chd",
                "chf",
                "stroke_hist"]

    def __init__(self, tYear=10):
        self.tYear = tYear

    def score(self, row):
        if row["female"]:
            return ukpdsom2_mi_female(row["AC"],
                                      row["diab_dur"],
                                      row["diab_age"],
                                      row["egfr"],
                                      row["hba1c"],
                                      row["chol_ldl_mmol"],
                                      row["albumin_urine"],
                                      row["pvd"],
                                      row["sbp"],
                                      row["cur_smoke"],
                                      row["wbc"],
                                      row["chf"],
                                      row["chd"],
                                      tYear=self.tYear)
        else:
            return ukpdsom2_mi_male(row["AC"],
                                    row["diab_dur"],
                                    row["diab_age"],
                                    row["EAsian"],
                                    row["hba1c"],
                                    row["chol_hdl_mmol"],
                                    row["chol_ldl_mmol"],
                                    row["albumin_urine"],
                                    row["pvd"],
                                    row["sbp"],
                                    row["cur_smoke"],
                                    row["wbc"],
                                    row["amp_hist"],
                                    row["chf"],
                                    row["chd"],
                                    row["stroke_hist"],
                                    tYear=self.tYear)

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["egfr_gt_60"] = row["egfr"]/10 if row["egfr"]<60 else 0
        feat_dict["ldl_mmol_10"] = row["chol_ldl_mmol"]*10
        feat_dict["hdl_mmol_10"] = row["chol_hdl_mmol"]*10
        feat_dict["ldl_35"] =  row["chol_ldl_mmol"]*10 if row["chol_ldl_mmol"] > 35 else 0
        feat_dict["mmalb"] = row["albumin_urine"] >= 50
        feat_dict["sbp_10"] = row["sbp"] / 10
        return feat_dict
