import numpy.testing as npt

from cvdm.score import ukpds, Ukpds


def test_ukpds():
    tmp = ukpds(45, 45, False, False, False, 7.5, 160, 4.9, tYear=20)
    npt.assert_almost_equal(tmp, 0.33, decimal=2)
    tmp = ukpds(55, 55, True, False, False, 6, 140, 4.0, tYear=10)
    npt.assert_almost_equal(tmp, 0.057, decimal=3)
    tmp = ukpds(55, 55, True, False, False, 8, 140, 4.0, tYear=10)
    npt.assert_almost_equal(tmp, 0.079, decimal=3)
    tmp = ukpds(55, 55, True, False, False, 10, 140, 4.0, tYear=10)
    npt.assert_almost_equal(tmp, 0.109, decimal=3)
    tmp = ukpds(55, 55, True, False, True, 10, 140, 4.0, tYear=10)
    npt.assert_almost_equal(tmp, 0.144, decimal=3)
    tmp = ukpds(55, 55, True, False, True, 10, 140, 8.0, tYear=10)
    npt.assert_almost_equal(tmp, 0.326, decimal=3)
    tmp = ukpds(55, 55, False, False, True, 10, 140, 8.0, tYear=10)
    npt.assert_almost_equal(tmp, 0.529, decimal=3)
    tmp = ukpds(55, 55, False, False, False, 8, 160, 8.0, tYear=10)
    npt.assert_almost_equal(tmp, 0.376, decimal=3)


def test_ukpds_json():
    uk = Ukpds(20)
    tmp = uk.score({"index_age": 45,
                    "female": False,
                    "diab_age": 45,
                    "race": "Asian",
                    "AC": False, 
                    "sbp":160,
                    "chol_tot":4.9,
                    "chol_hdl":1.0,
                    "tchdl": 4.9,
                    "hba1c":7.5,
                    "cur_smoke": False})
    npt.assert_almost_equal(tmp, 0.33, decimal=2)
    uk = Ukpds(10)
    tmp = uk.score({"index_age": 55,
                    "female": True,
                    "diab_age": 55,
                    "race": "Asian",
                    "AC": False,
                    "sbp":140,
                    "chol_tot":4.0,
                    "chol_hdl":1.0,
                    "tchdl": 4.0,
                    "hba1c":6,
                    "cur_smoke": False})
    npt.assert_almost_equal(tmp, 0.057, decimal=3)
    tmp = uk.score({"index_age": 55,
                    "female": True,
                    "diab_age": 55,
                    "race": "Caucasian or White",
                    "AC": False,
                    "sbp":140,
                    "chol_tot":4.0,
                    "chol_hdl":1.0,
                    "tchdl":4.0,
                    "hba1c":8,
                    "cur_smoke": False})
    npt.assert_almost_equal(tmp, 0.079, decimal=3)

