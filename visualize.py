import copy
import numpy as np

import open3d as o3d


class Visualize:
    def __init__(self):
        self.pcd = o3d.geometry.PointCloud()
        self.mesh = o3d.io.read_point_cloud("models/pcl.ply")
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Face landmarks Visualization', width=1920, height=1080)

    def update_mesh(self, pcl):
        self.pcd.points = o3d.utility.Vector3dVector(pcl)
        o3d.io.write_point_cloud("models/pcl.ply", self.pcd)
        self.mesh = o3d.io.read_point_cloud("models/pcl.ply")
        R = self.mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        self.mesh = self.mesh.rotate(R, center=(0, 0, 0))

    def show(self):
        self.vis.clear_geometries()
        self.vis.add_geometry(self.mesh)
        self.vis.poll_events()
        self.vis.update_renderer()
