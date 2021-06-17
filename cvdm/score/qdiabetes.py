"""
QDidabetes

See: https://qdiabetes.org/heart-failure/src.php

"""
import numpy as np

from cvdm.score import BaseRisk
from cvdm.score import clean_age, clean_hba1c, clean_bmi, clean_diab_dur
from cvdm.score import clean_tchdl, clean_bp

FEMALE_CCF = {
    "survival": np.array([0,
                          0.997221827507019,
                          0.994604170322418,
                          0.991730570793152,
                          0.988543510437012,
                          0.985216200351715,
                          0.981507956981659,
                          0.977895379066467,
                          0.974181711673737,
                          0.969876766204834,
                          0.964927375316620,
                          0.959847927093506,
                          0.954125642776489,
                          0.948961555957794,
                          0.942851066589355,
                          0.936939239501953]),
    "center": np.array([ 61.116180419921875,
                         0.322399437427521,
                         0.567802309989929,
                         2.584567546844482,
                        -1.227098584175110,
                         4.192868709564209,
                         0.847070097923279,
                         0.331943601369858,
                         0,
                         0,
                         0,
                         0]),
    "diabDur": np.array([0,
                         0.3734895516724550100000000,
                         0.4272400863870609000000000,
                         0.5445060839784665600000000,
                         0.6642326175899047100000000]),
    "ethnic": np.array([0,
                        0,
                        0.0121268111795051550000000,
                        0.0890841282777889020000000,
                        -0.0630035742862442990000000,
                        -0.0404902105074095640000000,
                        -0.3120678051325643200000000,
                        -0.0927498388096287120000000,
                        -0.2528100297781567500000000,
                        -0.1973637260842228100000000]),
    "smoke": np.array([0,
                       0.0967091332233832450000000,
                       0.2821028058053606800000000,
                       0.4215942284438172800000000,
                       0.5530645745533893100000000]),
    "beta": np.array([  0.0736700412014007770000000,
                       23.9487096235408750000000000,
                      -31.8537791574232610000000000,
                       -0.3252572926061894600000000,
                       -0.2171976293919814200000000,
                        0.0364104203967467660000000,
                       23.4410780063591200000000000,
                       10.4551225345768600000000000,
                        0.8874477721127933500000000,
                        0.6730365919558813900000000,
                        0.4923479055010733200000000,
                        0.3211215613581965800000000])
}

MALE_CCF = {
    "survival": np.array([0,
                          0.996442258358002,
                          0.993097066879272,
                          0.989732027053833,
                          0.985953390598297,
                          0.981611728668213,
                          0.977198719978333,
                          0.972434043884277,
                          0.967397689819336,
                          0.961909413337708,
                          0.956551969051361,
                          0.950842618942261,
                          0.943948566913605,
                          0.936877191066742,
                          0.928543448448181,
                          0.920696020126343]),
    "center": np.array([ 59.082637786865234,
                          0.113395698368549,
                          1.088435888290405,
                          2.458636045455933,
                         -1.105902791023254,
                          4.528943061828613,
                          0.326042562723160,
                          1.177061676979065,
                          0,
                          0,
                          0,
                          0]),
    "diabDur": np.array([0,
                         0.2511561621551418000000000,
                         0.3849557918896515700000000,
                         0.4013874319025841900000000,
                         0.5409884571186193100000000]),
    "ethnic": np.array([ 0,
                         0,
                        -0.0598115842149627780000000,
                        -0.0836282579919801760000000,
                         0.1318138372122178200000000,
                        -0.2584741955649905200000000,
                        -0.2588961117898153600000000,
                        -0.3878450613072409500000000,
                        -0.8990278720741535800000000,
                        -0.3065306136566966500000000]),
    "smoke": np.array([0,
                       0.0309188274582469360000000,
                       0.2901783652136286600000000,
                       0.2975987602015507900000000,
                       0.4131820986096286800000000]),
    "beta": np.array([0.0670714089231315420000000,
                      7.1099014255385811000000000,
                      2.9247387283215707000000000,
                      -0.2614980716613777300000000,
                      -0.1671878402926073600000000,
                      0.0277371549183472870000000,
                      -12.3361286873001800000000000,
                      21.4821781929691690000000000,
                      0.7273530624131829800000000,
                      0.7845250801959688900000000,
                      0.5737129078201175200000000,
                      0.1776151674711740600000000])
}


def _get_diab_dur_cat(diabDur):
    if diabDur < 1:
        return 0
    elif diabDur <= 3:
        return 1
    elif diabDur <= 6:
        return 2
    elif diabDur <= 10:
        return 3
    else:
        return 4


def _get_smoke_cat(heavy_smoke, moderate_smoke, light_smoke, prev_smoke):
    if heavy_smoke:
        return 4
    elif moderate_smoke:
        return 3
    elif light_smoke:
        return 2
    elif prev_smoke:
        return 1
    else:
        return 0


def _get_ethnic_cat(ac, easian):
    # caucasian, not recorded, indian, pakistani, bangladeish,
    # other asian, black carribean, black african, chinese, other
    if ac:
        return 7  # use black african
    elif easian:
        return 5  # use other asian
    else:
        # white or not stated
        return 0


def _frac_poly_female(bmi, hba1c, sbp):
    bmiDecile = bmi / 10
    bmi1 = np.power(bmiDecile, -1)
    bmi2 = np.power(bmiDecile, -0.5)
    hba1cDecade = hba1c / 100
    hba1c1 = np.power(hba1cDecade, -2)
    hba1c2 = np.power(hba1cDecade, -2)*np.log(hba1cDecade)
    sbpDecade = sbp/100
    sbp1 = np.power(sbpDecade, -0.5)
    sbp2 = np.log(sbpDecade)
    return bmi1, bmi2, hba1c1, hba1c2, sbp1, sbp2


def _frac_poly_male(bmi, hba1c, sbp):
    bmiDecile = bmi / 10
    bmi1 = np.power(bmiDecile, -2)
    bmi2 = np.log(bmiDecile)
    hba1cDecade = hba1c / 100
    hba1c1 = np.power(hba1cDecade, -2)
    hba1c2 = np.power(hba1cDecade, -2)*np.log(hba1cDecade)
    sbpDecade = sbp / 100
    sbp1 = np.log(sbpDecade)
    sbp2 = np.power(sbpDecade, 0.5)
    return bmi1, bmi2, hba1c1, hba1c2, sbp1, sbp2


def _survival(age, bmi, diabDur, ac, easian,
              hba1c, tchdl, sbp, heavy_smoke,
              moderate_smoke, light_smoke, prev_smoke,
              afib, cvd, renal, dmt1,
              genderInfo, tYear, fractalFunc):
    # apply the fractional polynomial transform
    bmi1, bmi2, hba1c1, hba1c2, sbp1, sbp2 = fractalFunc(bmi, hba1c, sbp)
    # construct the xfeat
    xFeat = np.array([age, bmi1, bmi2, hba1c1,
                      hba1c2, tchdl, sbp1, sbp2,
                      afib, cvd, renal, dmt1])
    # center the continuous variables
    xFeat = xFeat - genderInfo["center"]
    a = xFeat.dot(genderInfo["beta"])
    # add the conditional sums from the conditional
    a = a + genderInfo["diabDur"][_get_diab_dur_cat(diabDur)]
    a = a + genderInfo["smoke"][_get_smoke_cat(heavy_smoke, moderate_smoke, light_smoke, prev_smoke)]
    a = a + genderInfo["ethnic"][_get_ethnic_cat(ac, easian)]
    s = np.power(genderInfo["survival"][tYear], np.exp(a))
    return (1-s)


def qdiabetes(age, male, bmi, diab_dur, ac, easian,
              hba1c, tchdl, sbp, heavy_smoke,
              moderate_smoke, light_smoke, prev_smoke,
              afib, cvd, renal,
              tYear=5, dmt1=False):
    genderInfo = FEMALE_CCF
    fractalFunc = _frac_poly_female
    if male:
        genderInfo = MALE_CCF
        fractalFunc = _frac_poly_male
    return _survival(clean_age(age),
                     clean_bmi(bmi),
                     clean_diab_dur(diab_dur, 0),
                     ac,
                     easian,
                     clean_hba1c(hba1c, meas="mmol"),
                     clean_tchdl(tchdl),
                     clean_bp(sbp),
                     heavy_smoke, moderate_smoke,
                     light_smoke, prev_smoke, afib, cvd,
                     renal, dmt1, genderInfo, tYear, fractalFunc)


class QDiabetes(BaseRisk):
    tYear = None
    features = ["male",
                "index_age",
                "bmi",
                "hba1c_mmol",
                "tchdl",
                "sbp",
                "afib",
                "cvd_hist",
                "renal"]
    feat_key = features + ["diab_dur",
                           "AC", "EAsian", 
                           "heavy_smoke", 
                           "moderate_smoke", 
                           "light_smoke", 
                           "prev_smoke"]

    def __init__(self, tYear=5):
        self.tYear = tYear
    
    def score(self, row):
        return qdiabetes(row["index_age"],
                         row["male"],
                         row["bmi"],
                         row["diab_dur"],
                         row["AC"],
                         row["EAsian"],
                         row["hba1c_mmol"],
                         row["tchdl"],
                         row["sbp"],
                         row["heavy_smoke"],
                         row["moderate_smoke"],
                         row["light_smoke"],
                         row["prev_smoke"],
                         row["afib"],
                         row["cvd_hist"],
                         row["renal"],
                         self.tYear)

    def get_features(self, row):
        feat_dict = super().get_features(row)
        feat_dict["diab_dur_cat"] = _get_diab_dur_cat(row["diab_dur"])
        feat_dict["ethnic_cat"] = _get_ethnic_cat(row["AC"], row["EAsian"])
        feat_dict["smoke_cat"] = _get_smoke_cat(row["heavy_smoke"],
                                                row["moderate_smoke"],
                                                row["light_smoke"],
                                                row["prev_smoke"])
        return feat_dict



