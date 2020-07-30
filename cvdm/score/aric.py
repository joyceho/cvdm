"""
ARIC

See: https://aricnews.net/riskcalc/html/RC1.html

"""
import numpy as np

from cvdm.score import cox_surv, BaseRisk


# coefficients for survival
FEMALE_INFO = {
    "sm": 0.97507,
    "xBetaMed": 0.99852,
    "coef": np.array([-0.03301,  # age in years
                       0.00041,  # age^2
                       0.37361,  # race (white = 1, black = 0)
                       0.72618,  # tch 200-279 vs < 200
                       1.10225,  # tch >= 280 vs < 200
                       0.45810,  # hdl < 45 vs >= 50
                       0.55162,  # hdl 45-49 vs >= 50
                       0.01314,  # sbp in mmHg
                       0.48246,  # hypertension meds
                       0.78920,  # currently smoking
        ])
}

MALE_INFO = {
    "sm": 0.93299,
    "xBetaMed": 7.75110,
    "coef": np.array([ 0.22777,  # age in years
                      -0.00175,  # age^2
                       0.49310,  # race (white = 1, black = 0)
                       0.46038,  # tch 200-279 vs < 200
                       0.90874,  # tch >= 280 vs < 200
                       0.80719,  # hdl < 45 vs >= 50
                      -0.25732,  # hdl 45-49 vs >= 50
                       0.00437,  # sbp in mmHg
                      -0.05461,  # hypertension meds
                       0.15208,  # currently smoking
        ])
}


def aric(age, isMale, isCauc, tc, hdl, sbp, htn, smoke):
    # set the coeff based on male or female
    genderInfo = FEMALE_INFO
    if isMale:
        genderInfo = MALE_INFO
    xFeat = np.array([age, age**2, isCauc,
                     tc >= 200 and tc <= 279,
                     tc >= 280,
                     hdl < 45,
                     hdl >= 45 and hdl <= 49,
                     sbp, htn, smoke])
    return cox_surv(xFeat, genderInfo["coef"],
                    genderInfo["sm"],
                    genderInfo["xBetaMed"])


class Aric(BaseRisk):
    features = ["index_age",
                "male",
                "Cauc",
                "sbp",
                "htn_treat",
                "cur_smoke"]
    feat_key = features + ["chol_tot", "chol_hdl"]

    def score(self, row):
        return aric(row["index_age"],
                    row["male"],
                    row["Cauc"],
                    row["chol_tot"],
                    row["chol_hdl"],
                    row["sbp"],
                    row["htn_treat"],
                    row["cur_smoke"])

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["age_sq"] = row["index_age"]**2
        feat_dict["tc_279"] = row["chol_tot"] >= 200 and row["chol_tot"] <= 279
        feat_dict["tc_280"] = row["chol_tot"] >= 280
        feat_dict["hdl_45"] = row["chol_hdl"] < 45
        feat_dict["hdl_49"] = row["chol_hdl"] >= 45 and row["chol_hdl"] <= 49
        return feat_dict
