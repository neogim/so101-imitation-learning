import unittest
from argparse import Namespace
from pathlib import Path
from phase8_orientation_dispatch import orientation_from_quaternion, command_for_strategy

class Phase8DispatchTests(unittest.TestCase):
    def test_quaternion_orientation(self):
        self.assertEqual(orientation_from_quaternion([1,0,0,0])[0], 'upright')
        self.assertEqual(orientation_from_quaternion([0.70710678,0,0.70710678,0])[0], 'horizontal')
        self.assertEqual(orientation_from_quaternion([0.9238795,0,0.3826834,0])[0], 'unsupported_tilt')
        self.assertEqual(orientation_from_quaternion([0,1,0,0])[0], 'upright')
        with self.assertRaises(ValueError): orientation_from_quaternion([0,0,0,0])
    def test_horizontal_command(self):
        a=Namespace(yaw_deg=90, results_root=Path('/tmp/results'),csv=Path('/tmp/data.csv'),
            mapping=Path('/tmp/mapping.json'),urdf=Path('/tmp/a.urdf'),out=Path('/tmp/out'),
            wrist_camera=True,dataset_root=None)
        cmd=command_for_strategy(a,'horizontal')
        self.assertIn('90',cmd)
        self.assertIn('--wrist-camera',cmd)
        a.yaw_deg=91
        with self.assertRaises(ValueError):command_for_strategy(a,'horizontal')
    def test_upright_command(self):
        a=Namespace(dataset_root=Path('/tmp/data'),urdf=Path('/tmp/a.urdf'),out=Path('/tmp/out'),
                    wrist_camera=True)
        cmd=command_for_strategy(a,'upright')
        self.assertIn('--depth-only',cmd)
        self.assertIn('--wrist-camera',cmd)

if __name__ == '__main__': unittest.main()
