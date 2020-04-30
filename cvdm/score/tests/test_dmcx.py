import numpy.testing as npt

from cvdm.score import dmcx, Dmcx


def test_dmcx():
    tmp = dmcx(55, 80, 4.3, 2.0, False, 5, True,
               120, 60, 7, False, 28, False, False)
    npt.assert_almost_equal(tmp, 0.03, decimal=2)
    tmp = dmcx(55, 80, 4.3, 2.0, False, 5, False,
               120, 60, 7, False, 28, False, False)
    npt.assert_almost_equal(tmp, 0.06, decimal=2)
    tmp = dmcx(65, 80, 4.3, 4.0, True, 5, True,
               200, 80, 9, True, 30, True, True)
    npt.assert_almost_equal(tmp, 0.28, decimal=2)
    tmp = dmcx(65, 80, 4.3, 4.0, True, 5, False,
               200, 80, 9, True, 30, True, True)
    npt.assert_almost_equal(tmp, 0.38, decimal=2)


def test_dmcx_json():
    dm = Dmcx()
    tmp = dm.score({"index_age": 55,
                    "female": True,
                    "cur_smoke": False,
                    "hba1c": 7,
                    "bmi": 28,
                    "sbp":120,
                    "dbp":60,
                    "egfr": 80,
                    "tchdl": 4.3,
                    "albumin_creat_mgmmol": 2.0,
                    "diab_dur": 5,
                    "htn_treat": False,
                    "insulin": False,
                    "a_glucose": False})
    npt.assert_almost_equal(tmp, 0.03, decimal=2)
