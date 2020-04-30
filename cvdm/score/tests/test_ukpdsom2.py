import numpy.testing as npt

from cvdm.score import ukpdsom2_chf, UkpdsOM2CHF


def test_ukpdsom2_chf():
    tmp = ukpdsom2_chf(8, 62, False, 32, 50, 3.0,
                       55, False, True, False, tYear=1)
    npt.assert_almost_equal(tmp, 0.027, decimal=3)


def test_chf():
    model = UkpdsOM2CHF(tYear=1)
    tmp = model.score({"diab_dur": 8,
                       "diab_age": 62,
                       "afib": False,
                       "bmi": 32,
                       "egfr": 50,
                       "chol_ldl_mmol": 3.0,
                       "albumin_urine": 55,
                       "pvd": False,
                       "amp_hist": True,
                       "ulcer_hist": False})
    npt.assert_almost_equal(tmp, 0.027, decimal=3)

