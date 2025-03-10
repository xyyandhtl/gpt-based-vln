import numpy as np
import open3d as o3d

"""
This file contains a workaround LineMesh class for Open3D which is a lineset with cylinders instead of lines.
This is useful for visualizing lines of different thicknesses in Open3D.

From:
    https://github.com/isl-org/Open3D/pull/738#issuecomment-564785941
    https://github.com/isl-org/Open3D/pull/738#issuecomment-697027818
"""


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines, colors=[0, 1, 0], radius=0.05):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(lines)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length
            )
            cylinder_segment = cylinder_segment.translate(translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a),
                    center=cylinder_segment.get_center(),
                )
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


##########################################################################################

from tag_mapping.utils import get_box_corners


def box_to_linemesh(box, color=(0, 1, 0), radius=0.02):
    """
    Get a LineMesh from a box type.

    The box type must be supported by get_box_corners() as we assume that
    get_box_corners() will return the corners in the expected order
    """
    box_points = get_box_corners(box)

    box_lines = np.array(
        [
            [0, 1],
            [0, 3],
            [1, 2],
            [2, 3],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 7],
            [5, 6],
            [6, 7],
        ]
    )

    return LineMesh(
        points=box_points,
        lines=box_lines,
        colors=color,
        radius=radius,
    )
