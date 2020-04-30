import numpy.testing as npt

from cvdm.score import frs_primary, frs_simple, FrsPrimary, FrsSimple


def test_frs_primary():
    tmp = frs_primary(True, 61, 180, 47, 124, False, True, False)
    npt.assert_almost_equal(tmp, 0.1048, decimal=4)
    tmp = frs_primary(False, 53, 161, 55, 125, True, False, True)
    npt.assert_almost_equal(tmp, 0.1562, decimal=4)


def test_frs_primary_json():
    frs = FrsPrimary()
    tmp = frs.score({"female": True,
                     "index_age": 61,
                     "chol_tot": 180,
                     "chol_hdl": 47,
                     "sbp": 124,
                     "htn_treat": False,
                     "dm": False,
                     "cur_smoke": True})
    npt.assert_almost_equal(tmp, 0.1048, decimal=4)
    tmp = frs.score({"female": False,
                     "index_age": 53,
                     "chol_tot": 161,
                     "chol_hdl": 55,
                     "sbp": 125,
                     "htn_treat": True,
                     "dm": True,
                     "cur_smoke": False})
    npt.assert_almost_equal(tmp, 0.1562, decimal=4)


def test_frs_simple():
    score = frs_simple("F", 35, 24.3, 122, False, True, False)
    npt.assert_almost_equal(score, 0.029352227213368165, decimal=5)


def test_frs_simple_json():
    frs = FrsSimple()
    tmp = frs.score({"female": True,
                     "index_age": 35,
                     "height": 72.0,
                     "weight": 190.0,
                     "bmi": 24.3,
                     "sbp": 122,
                     "htn_treat": False,
                     "dm": False,
                     "cur_smoke": True})
    npt.assert_almost_equal(tmp, 0.029352227213368165, decimal=5)
