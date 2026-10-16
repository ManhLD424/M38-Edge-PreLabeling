from pathlib import Path


def test_fingerprinted_vendor_text_has_canonical_lf_for_linux_deploy():
    for path in Path('docs/vendor').iterdir():
        if path.suffix in ('.txt','.js','.mjs'):
            assert b'\r\n' not in path.read_bytes(), f'{path} changes hash when checked out on Linux'
