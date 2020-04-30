import numpy.testing as npt

from cvdm.score import ndr, Ndr


def test_ndr():
    tmp = ndr(53, 5, 4.3, 8, 150, 32,
              True, False, True, False, False, False)
    npt.assert_almost_equal(tmp, 0.109, decimal=3)


def test_ndr_json():
    ndr = Ndr()
    tmp = ndr.score({"diab_age": 53,
                    "diab_dur": 5,
                    "chol_tot": 4.3,
                    "chol_hdl": 1,
                    "tchdl": 4.3,
                    "hba1c": 8,
                    "sbp": 150,
                    "height": 64.6,
                    "weight": 190.0,
                    "bmi": 32,
                    "male": True,
                    "cur_smoke": False,
                    "microalbum": True,
                    "macroalbum": False,
                    "albumin_urine": [35],
                    "afib": False,
                    "cvd_hist": False})
    npt.assert_almost_equal(tmp, 0.109, decimal=3)
