import json
import numpy as np
import laspy
from data.raw_cloud import sample_cloud, open_coordinates
from utils.visualize import export_full_scene_ply, FULL_VERTEX_DTYPE


def test_las_sampling_scaling_and_full_export(tmp_path):
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.scales = [.01,.02,.03]
    header.offsets = [443000.,4360000.,1200.]
    cloud = laspy.LasData(header)
    cloud.x = 443000.+np.arange(20)*.25
    cloud.y = 4360000.+np.arange(20)*.5
    cloud.z = 1200.+np.arange(20)*.3
    cloud.red = np.full(20,65535,dtype=np.uint16)
    path = tmp_path/'scene.las'; cloud.write(path)
    xyz,color,indices,count,origin = sample_cloud(path,7,3)
    assert count==20 and len(indices)==7 and np.all(color[:,0]==1)
    expected = np.column_stack((cloud.x,cloud.y,cloud.z))
    np.testing.assert_allclose(xyz,expected[indices])
    with open_coordinates(path) as coords:
        np.testing.assert_allclose(coords[2:6],expected[2:6])
    np.save(tmp_path/'coord.npy',(xyz-origin).astype(np.float32))
    np.save(tmp_path/'source_index.npy',indices)
    (tmp_path/'source.json').write_text(json.dumps(dict(path=str(path),origin=origin.tolist(),original_count=count)))
    report=export_full_scene_ply(tmp_path/'full.ply',tmp_path,np.linspace(0,1,7),chunk_size=3)
    assert report['point_count']==20
    with (tmp_path/'full.ply').open('rb') as f:
        while f.readline()!=b'end_header\n': pass
        vertices=np.fromfile(f,dtype=FULL_VERTEX_DTYPE)
    np.testing.assert_allclose(np.column_stack([vertices[k] for k in ('x','y','z')]),expected)
    np.testing.assert_allclose(vertices['probability'][indices],np.linspace(0,1,7))
