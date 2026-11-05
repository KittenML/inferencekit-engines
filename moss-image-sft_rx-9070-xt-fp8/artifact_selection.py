"""Select only artifact paths already individually verified by the installer."""
from pathlib import Path


def select_verified(paths):
    paths = [Path(path) for path in paths]
    if len({path.name for path in paths}) != len(paths):
        raise ValueError('duplicate artifact filename')
    wheels = sorted(path for path in paths if path.name.endswith('.whl'))
    sdists = sorted(path for path in paths if path.name.endswith('.tar.gz'))
    if len(wheels) + len(sdists) != len(paths):
        raise ValueError('unsupported artifact extension')
    if not wheels:
        raise ValueError('no verified wheels')
    return wheels, sdists
