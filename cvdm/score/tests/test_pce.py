import numpy.testing as npt

from cvdm.score import pce, Pce


def test_pce():
    tmp = pce(False, False, 60, 150, 65, 120, False, False, True, 10)
    npt.assert_almost_equal(tmp, 0.093, decimal=3)
    tmp = pce(True, False, 60, 150, 65, 120, False, False, True, 10)
    npt.assert_almost_equal(tmp, 0.040, decimal=3)
    tmp = pce(True, True, 60, 150, 65, 120, False, False, True, 10)
    npt.assert_almost_equal(tmp, 0.070, decimal=3)
    tmp = pce(False, True, 60, 150, 65, 120, False, False, True, 10)
    npt.assert_almost_equal(tmp, 0.115, decimal=3)
    tmp = pce(False, True, 60, 150, 65, 120, True, True, True, 10)
    npt.assert_almost_equal(tmp, 0.298, decimal=3)


def test_pce_json():
    pce = Pce(risk=10)
    tmp = pce.score({"female": False,
                     "AC": False,
                     "index_age": 60,
                     "chol_tot": 150,
                     "chol_hdl": 65,
                     "sbp": 120,
                     "cur_smoke": False,
                     "dm": True,
                     "htn_treat": False})
    npt.assert_almost_equal(tmp, 0.093, decimal=3)
