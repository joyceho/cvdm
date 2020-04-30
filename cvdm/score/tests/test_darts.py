import numpy.testing as npt

from cvdm.score import darts, Darts

# diabAge, diabDur, cholTot, prevSmoke,
#           curSmoke, isMale, hba1c, follow5, sbp,
#           htn, height


def test_darts():
    tmp = darts(59, 6, 5.8, 0, 1, 1,
                8, 1, 160, 0, 1.7, 5)
    npt.assert_almost_equal(tmp, 0.54, decimal=2)


def test_darts_class():
    model = Darts(5)
    tmp = model.score({"diab_age": 59,
                       "diab_dur": 6,
                       "chol_tot_mmol": 5.8,
                       "prev_smoke": False,
                       "cur_smoke": True,
                       "male": True,
                       "hba1c": 8,
                       "5y_follow": True,
                       "sbp": 160,
                       "htn_treat": False,
                       "height_m": 1.7})
    npt.assert_almost_equal(tmp, 0.54, decimal=2)
