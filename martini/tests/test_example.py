"""Smoke tests: verify top-level package exports are importable."""


def test_compute_ddlr_importable():
    from .. import compute_ddlr
    assert callable(compute_ddlr)


def test_roman_pixel_scale_importable():
    from .. import ROMAN_PIXEL_SCALE
    assert ROMAN_PIXEL_SCALE > 0
