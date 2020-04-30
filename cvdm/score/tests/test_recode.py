import numpy.testing as npt

from cvdm.score import recode, Recode

           
def test_recode():
    risk = recode(60, False, False,
                  False, 140, False, False, False,
                  False, 8, 190, 50, 1.1, 10)
    npt.assert_almost_equal(risk, 0.03, decimal=2)
    risk = recode(70, False, False,
                  False, 140, False, False, False,
                  False, 8, 190, 50, 1.1, 10)
    npt.assert_almost_equal(risk, 0.05, decimal=2)
    risk = recode(70, True, False,
                  True, 140, False, True, False,
                  False, 8, 190, 50, 1.1, 10)
    npt.assert_almost_equal(risk, 0.14, decimal=2)
    risk = recode(75, True, False,
                  True, 140, True, True, True,
                  False, 8, 190, 50, 1.1, 10)
    npt.assert_almost_equal(risk, 0.39, decimal=2)
    risk = recode(60, True, True,
                  True, 140, True, True, True,
                  True, 8, 190, 50, 1.1, 10)
    npt.assert_almost_equal(risk, 0.36, decimal=2)


def test_recode_json():
    rc = Recode()
    risk = rc.score({"index_age": 60,
                     "female": False,
                     "AC": False,
                     "cur_smoke": False,
                     "sbp":140,
                     "cvd_hist": False,
                     "bpld": False,
                     "statin": False,
                     "anticoagulant": False,
                     "hba1c": 8,
                     "chol_tot": 190,
                     "chol_hdl": 50,
                     "creat": 1.1,
                     "albumin_creat": 10})
    npt.assert_almost_equal(risk, 0.03, decimal=2)
