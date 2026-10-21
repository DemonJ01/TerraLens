def test_version_exists():
    import terralens

    assert hasattr(terralens, "__version__")
