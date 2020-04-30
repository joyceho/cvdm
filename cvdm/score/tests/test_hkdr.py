import numpy.testing as npt

from cvdm.score import hkdr_chd, HkdrCHD
from cvdm.score import hkdr_hf, HkdrHF


def test_hkdr_chd():
    tmp = hkdr_chd(59, True, False, 5, 105, 2.3, 3.87)
    npt.assert_almost_equal(tmp, 0.082, decimal=3)


def test_hkdr_chd_json():
    chd = HkdrCHD()
    tmp = chd.score({"index_age": 59,
                     "female": True,
                     "cur_smoke": False,
                     "diab_dur": 5,
                     "egfr": 105,
                     "albumin_creat_mgmmol": 2.3,
                     "nonhdl_mmoll": 3.87})
    npt.assert_almost_equal(tmp, 0.082, decimal=3)


def test_hkdr_hf():
    tmp = hkdr_hf(False, 59, 32, 8, 2.5, 13.8, True)
    npt.assert_almost_equal(tmp, 0.038, decimal=3)
    tmp = hkdr_hf(True, 59, 32, 8, 2.5, 13.8, True)
    npt.assert_almost_equal(tmp, 0.064, decimal=3)
    tmp = hkdr_hf(False, 59, 24.3, 8, 2.5, 13.8, True)
    npt.assert_almost_equal(tmp, 0.024, decimal=3)


def test_hkdr_hf_json():
    hf = HkdrHF()
    tmp = hf.score({"index_age": 59,
                    "female": False,
                    "albumin_creat_mgmmol":2.5,
                    "bmi": 24.3,
                    "hba1c": 8,
                    "hb": 13.8,
                    "chd": True})
    npt.assert_almost_equal(tmp, 0.024, decimal=3)
