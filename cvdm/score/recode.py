"""
RECODE risk calculation
"""
import argparse
import json
import numpy as np
import tqdm

from cvdm.score import cox_surv, BaseRisk


## Coefficients for CHF Recode
CHD_INFO ={
    "coef": np.array([ 0.05268, # age (year)
                       0.25290, # female
                       -0.04969, # black ethnicity
                       0.29050, # smoking
                       0.00121, # SBP
                       1.00700, # history of cvd
                       0.63890, # blood pressure lowering drugs
                       -0.11750, # statins
                       0.73650, # anticoagulants
                       0.20920, # hba1c %
                       -0.00136, # total cholesterol
                       -0.01758, # hdl cholesterol
                       0.82104, # serum creatine mg/dl
                       0.00041  # urine albumin:creatine ratio
    ]),
    "s0": 0.96,      # lambda in the equation
    "const": 5.15    # the mean to subtract by
}


MI_INFO ={
    "coef": np.array([ 0.04363, # age (year)
                      -0.20660, # female
                      -0.11630, # black ethnicity
                       0.23580, # smoking
                      -0.00514, # SBP
                       0.96180, # history of cvd
                      -0.12480, # blood pressure lowering drugs
                       0.04699, # statins
                       0.54400, # anticoagulants
                       0.21350, # hba1c %
                       0.00019, # total cholesterol
                      -0.01358, # hdl cholesterol
                       0.08027, # serum creatine mg/dl
                       0.00042  # urine albumin:creatine ratio
    ]),
    "s0": 0.96,      # lambda in the equation
    "const": 5.15    # the mean to subtract by
}


STROKE_INFO ={
    "coef": np.array([ 0.02896, # age (year)
                      -0.00326, # female
                       0.27160, # black ethnicity
                       0.16650, # smoking
                       0.01659, # SBP
                       0.41380, # history of cvd
                       0.15980, # blood pressure lowering drugs
                      -0.18870, # statins
                      -0.13870, # anticoagulants
                       0.33650, # hba1c %
                       0.00171, # total cholesterol
                      -0.00639, # hdl cholesterol
                       0.59550, # serum creatine mg/dl
                       0.00030  # urine albumin:creatine ratio
    ]),
    "s0": 0.96,      # lambda in the equation
    "const": 5.15    # the mean to subtract by
}


def recode(age, isFemale, ethnicity, smoking, sbp, 
           cvdHist, bpld, statin, anticoag,
           hba1c, tchol, hdl, creat, albumcreat,
           target="CHF"):
    coefInfo = CHD_INFO
    if target == "MI":
        coefInfo = MI_INFO
    if target == "STROKE":
        coefInfo = STROKE_INFO
    """
    Calculate the survival value
    """
    xFeat = np.array([age, isFemale, ethnicity,
                  smoking, sbp, cvdHist,
                  bpld, statin, anticoag,
                  hba1c, tchol, hdl, creat,
                  albumcreat])
    return cox_surv(xFeat, coefInfo["coef"],
                    coefInfo["s0"], coefInfo["const"])


class Recode(BaseRisk):
    target = None
    features = ["index_age",
                "female",
                "AC",
                "cur_smoke",
                "sbp", 
                "cvd_hist",
                "bpld", 
                "statin",
                "anticoagulant", 
                "hba1c",
                "chol_tot", 
                "chol_hdl",
                "creat", 
                "albumin_creat"]

    def __init__(self, target="CHF"):
        self.target = target
    
    def score(self, row):
        return recode(row["index_age"],
                      row["female"],
                      row["AC"],
                      row["cur_smoke"],
                      row["sbp"], 
                      row["cvd_hist"],
                      row["bpld"], 
                      row["statin"],
                      row["anticoagulant"], 
                      row["hba1c"],
                      row["chol_tot"], 
                      row["chol_hdl"],
                      row["creat"], 
                      row["albumin_creat"],
                      target=self.target)
