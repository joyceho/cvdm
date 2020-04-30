import numpy.testing as npt

from cvdm.score import fremantle, Fremantle


def test_fremantle():
    tmp = fremantle(59, True, True, 8, 0.92, 0.79, True, False)
    npt.assert_almost_equal(tmp, 0.062, decimal=3)


def test_fremantle_json():
    fr = Fremantle()
    tmp = fr.score({"index_age": 59,
                     "male": True,
                     "cvd_hist": True,
                     "hba1c": 8,
                     "albumin_creat_mgmmol":0.92,
                     "chol_hdl_mmol": 0.79,
                     "SEuro": True,
                     "Abor": False})
    npt.assert_almost_equal(tmp, 0.062, decimal=3)
