def test_version_is_string():
    from martini import __version__
    assert isinstance(__version__, str)
