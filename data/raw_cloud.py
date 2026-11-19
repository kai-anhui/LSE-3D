"""Read indexed samples without loading a multi-gigabyte LAS into RAM."""
from pathlib import Path, PureWindowsPath
import numpy as np
from contextlib import contextmanager


def resolve_source_path(recorded_path, scene_dir, override=None):
    """Resolve an imported scene after moving between Windows/Linux machines.

    An explicit --source wins. Otherwise try the stored path, then the known
    datasets_raw/point_data layout near the scene or working directory. Never
    silently choose among multiple files; export validates count/coordinates too.
    """
    if override is not None:
        source = Path(override).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"--source does not exist: {source}")
        return source.resolve()
    source = Path(recorded_path).expanduser()
    if source.is_file():
        return source.resolve()
    # PureWindowsPath also recognizes '/' separators, independent of host OS.
    name = PureWindowsPath(str(recorded_path)).name
    base = Path(scene_dir).resolve()
    roots = (base, *base.parents, Path.cwd().resolve())
    candidates = set()
    for root in roots:
        for directory in (root / "datasets_raw" / "point_data", root / "point_data"):
            candidate = directory / name
            if candidate.is_file():
                candidates.add(candidate.resolve())
    if len(candidates) == 1:
        return candidates.pop()
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple source files match {name}; specify --source: "
            + ", ".join(str(p) for p in sorted(candidates))
        )
    raise FileNotFoundError(
        f"Original cloud not found: {recorded_path}. Copy the original file into "
        f"datasets_raw/point_data/{name}, or specify --source with its local path."
    )


def las_xyz(points, header):
    return np.column_stack((points.X, points.Y, points.Z)).astype(np.float64) * header.scales + header.offsets


class LasCoordinates:
    """Slice-only XYZ view: no full-scene coordinate materialization."""
    def __init__(self, cloud):
        self.cloud = cloud

    def __len__(self):
        return self.cloud.header.point_count

    def __getitem__(self, key):
        return las_xyz(self.cloud.points[key], self.cloud.header)


@contextmanager
def open_coordinates(path):
    path = Path(path)
    if path.suffix.lower() == '.las':
        import laspy
        with laspy.mmap(str(path)) as cloud:
            yield LasCoordinates(cloud)
    elif path.suffix.lower() in ('.txt', '.xyz'):
        yield np.loadtxt(path, dtype=np.float64, usecols=(0,1,2), encoding='utf-8-sig', ndmin=2)
    else:
        raise ValueError('Supported raw formats: LAS, whitespace TXT/XYZ')


def sample_cloud(path, max_points, seed):
    path = Path(path)
    if max_points <= 0:
        raise ValueError("Multi-scene preparation requires a positive sample budget")
    rng = np.random.default_rng(seed)
    if path.suffix.lower() == '.las':
        import laspy
        with laspy.mmap(str(path)) as cloud:
            count = cloud.header.point_count
            origin = (cloud.header.mins + cloud.header.maxs) / 2
            indices = np.sort(rng.choice(count, min(count, max_points), replace=False))
            selected = cloud.points[indices].copy()
            xyz = las_xyz(selected, cloud.header)
            names = set(selected.point_format.dimension_names)
            color = np.column_stack((selected.red, selected.green, selected.blue)).astype(np.float32) if {'red','green','blue'} <= names else np.zeros_like(xyz, dtype=np.float32)
    elif path.suffix.lower() in ('.txt', '.xyz'):
        cloud = np.loadtxt(path, dtype=np.float64, encoding='utf-8-sig', ndmin=2)
        count = len(cloud)
        origin = (cloud[:,:3].min(0) + cloud[:,:3].max(0))/2
        indices = np.sort(rng.choice(count, min(count, max_points), replace=False))
        xyz = cloud[indices,:3]
        color = cloud[indices,3:6].astype(np.float32) if cloud.shape[1]>=6 else np.zeros_like(xyz,dtype=np.float32)
    else:
        raise ValueError(f'Unsupported raw point cloud: {path.suffix}')
    color /= 65535 if color.max()>255 else 255 if color.max()>1 else 1
    return xyz, color, indices, int(count), origin
