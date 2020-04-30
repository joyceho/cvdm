import numpy.testing as npt

from cvdm.score import aric, Aric


def test_aric():
    tmp = aric(53, False, False, 190, 50, 140, False, False)
    npt.assert_almost_equal(tmp, 0.03169, decimal=5)
    tmp = aric(53, False, True, 190, 50, 140, False, False)
    npt.assert_almost_equal(tmp, 0.04571, decimal=5)
    tmp = aric(60, True, True, 220, 40, 140, True, False)
    npt.assert_almost_equal(tmp, 0.38077, decimal=5)


def test_aric_json():
    ar = Aric()
    tmp = ar.score({"male": False,
                    "index_age": 53,
                    "Cauc": False,
                    "chol_tot": 190,
                    "chol_hdl": 50,
                    "sbp": 140,
                    "htn_treat": False,
                    "cur_smoke": False})
    npt.assert_almost_equal(tmp, 0.03169, decimal=5)
