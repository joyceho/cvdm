import numpy.testing as npt

from cvdm.score import advance, Advance


def test_advance():
    tmp = advance(50, False, 3, 50, True, True, 7, 50, 3.3, True)
    npt.assert_almost_equal(tmp, 0.062, decimal=3)


def test_advance_json():
    ad = Advance()
    tmp = ad.score({"diab_age": 50,
                    "female": False, 
                    "diab_dur": 3,
                    "pp": 50,
                    "retinopathy": True,
                    "afib": True,
                    "hba1c": 7,
                    "albumin_creat": 50,
                    "nonhdl_mmol": 3.3,
                    "htn_treat": True})
    npt.assert_almost_equal(tmp, 0.062, decimal=3)
