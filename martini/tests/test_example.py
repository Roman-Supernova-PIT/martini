"""Smoke tests: verify top-level package exports are importable and callable."""


def test_compute_offset_angle_importable():
    from .. import compute_offset_angle
    assert callable(compute_offset_angle)


def test_compute_dlr_importable():
    from .. import compute_dlr
    assert callable(compute_dlr)


def test_find_host_importable():
    from .. import find_host
    assert callable(find_host)
