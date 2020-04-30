import numpy.testing as npt

from cvdm.score import dcs, Dcs


def test_dcs():
    tmp = dcs(55, False, False, False, 8, 120,
              False, False, False, False, True,
              4.3, False, False, 5, False)
    npt.assert_almost_equal(tmp, 0.172, decimal=3)
    tmp = dcs(55, False, False, False, 8, 120,
              False, False, False, False, True,
              4.3, False, False, 5, False, target="MI")
    npt.assert_almost_equal(tmp, 0.071, decimal=3)
    tmp = dcs(55, True, False, False, 8, 120,
              False, False, False, False, True,
              4.3, False, False, 5, False)
    npt.assert_almost_equal(tmp, 0.147, decimal=3)
    tmp = dcs(55, True, False, False, 8, 120,
              False, False, False, False, True,
              4.3, True, False, 5, False)
    npt.assert_almost_equal(tmp, 0.175, decimal=3)
    tmp = dcs(55, True, False, False, 8, 120,
              False, False, False, False, True,
              4.3, True, False, 5, False, target="MI")
    npt.assert_almost_equal(tmp, 0.065, decimal=3)
    

def test_dcs_json():
    cvd = Dcs("CVD")
    tmp = cvd.score({"diab_age": 55,
                      "female": False,
                      "prev_smoke": False,
                      "cur_smoke": False,
                      "hba1c": 8,
                      "sbp": 120,
                      "Maori": False,
                      "EAsian": False,
                      "Pacific": False,
                      "IndoAsian": False,
                      "ODcs": True,
                      "tchdl": 4.3,
                      "microalbum": False,
                      "macroalbum": False,
                      "diab_dur": 5,
                      "htn_treat": False})
    npt.assert_almost_equal(tmp, 0.172, decimal=3)
    mi = Dcs("MI")
    tmp = mi.score({"diab_age": 55,
                     "female": False,
                     "prev_smoke": False,
                     "cur_smoke": False,
                     "hba1c": 8,
                     "sbp": 120,
                     "Maori": False,
                     "EAsian": False,
                     "Pacific": False,
                     "IndoAsian": False,
                     "ODcs": True,
                     "microalbum": False,
                     "macroalbum": False,
                     "tchdl": 4.3,
                     "diab_dur": 5,
                     "htn_treat": False})
    npt.assert_almost_equal(tmp, 0.071, decimal=3)
    tmp = cvd.score({"diab_age": 55,
                      "female": True,
                      "prev_smoke": False,
                      "cur_smoke": False,
                      "hba1c": 8,
                      "sbp": 120,
                      "Maori": False,
                      "EAsian": False,
                      "Pacific": False,
                      "IndoAsian": False,
                      "ODcs": True,
                      "tchdl": 4.3,
                      "microalbum": False,
                      "macroalbum": False,
                      "diab_dur": 5,
                      "htn_treat": False})
    npt.assert_almost_equal(tmp, 0.147, decimal=3)
    tmp = cvd.score({"diab_age": 55,
                      "female": True,
                      "prev_smoke": False,
                      "cur_smoke": False,
                      "hba1c": 8,
                      "sbp": 120,
                      "Maori": False,
                      "EAsian": False,
                      "Pacific": False,
                      "IndoAsian": False,
                      "ODcs": True,
                      "tchdl": 4.3,
                      "microalbum": True,
                      "macroalbum": False,
                      "diab_dur": 5,
                      "htn_treat": False})
    npt.assert_almost_equal(tmp, 0.175, decimal=3)
