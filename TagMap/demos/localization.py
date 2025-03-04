import os

import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
from matplotlib import cm

# import ipywidgets as widgets
# # from IPython.display import display

from tag_mapping.datasets.matterport import (
    read_matterport_image_file,
    read_matterport_depth_file,
    MatterportFilenameBridge
)

from tag_mapping import TagMap

from tag_mapping.localization import tagmap_entries_to_viewpoints, localization_pipeline

from tag_mapping.utils import box_to_linemesh

# Load scene data
# Please first download the demo data by running download_demo_data.sh.

scene_dir = '../datasets/demo_data'
tag_map = TagMap.load(f'{scene_dir}/scene.tagmap')
intrinsics = tag_map.metadata["intrinsics"]
images_dir = os.path.join(scene_dir, 'color')
depths_dir = os.path.join(scene_dir, 'depth')
poses_dir = os.path.join(scene_dir, 'poses')
mesh_path = os.path.join(scene_dir, 'mesh.ply')

# Load and visualize the mesh
scene_mesh = o3d.io.read_triangle_mesh(mesh_path)
o3d.visualization.draw_geometries([scene_mesh])

# Localize a selected tag
# Select a tag recognized in the scene to localize
options = sorted(list(tag_map.unique_objects))
print(f'options:', options)
# query_dropdown = widgets.Dropdown(options=options, description='Select an tag:')
# display(query_dropdown)

# Retrieve corresponding viewpoints for the selected tag.
#
# Rerun this block after changing the selection
query_entries = tag_map.query('bed')

# Show the images for a few of the viewpoints corresponding to the tag
max_show = 6
num_show = min(len(query_entries), max_show)

fig, axes = plt.subplots(1, num_show, figsize=(3 * num_show, 6))

for i in range(num_show):
    entry = query_entries[i]
    image_filename = entry.extras['image_filename']
    conf = entry.extras['confidence']

    image = read_matterport_image_file(
        os.path.join(images_dir, image_filename))

    try:
        ax = axes[i]
    except TypeError:
        ax = axes

    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'confidence: {conf:.2f}')
    ax.set_aspect(1)

plt.show()

# Compute coarse-grained localizations in 3D for the selected tag
# For each viewpoint corresponding to the selected tag, we first get their frustums in 3D.
viewpoints = tagmap_entries_to_viewpoints(
    entries=query_entries,
    intrinsics=intrinsics,

    # set the near plane of the viewpoint frustum to a constant distance away
    near_dist_fn=lambda x: 0.2,

    # the far plane of the viewpoint frustum is set as the 80th percentile depth value
    # of each viewpoint
    far_dist_fn=lambda entry: entry.extras['depth_percentiles']['0.8'],
)

# Visualize the retrieved viewpoint frustums
o3d.visualization.draw_geometries([scene_mesh] + [vp.o3d_lineset(color=np.random.rand(3)) for vp in viewpoints])

# Localization pipeline
# The localization pipeline takes as input the frustums of the retrieved viewpoints and performs a voting procedure over voxels in the scene to generate localized regions for the selected tag.
#
# The final output is a set of proposed localizations for the tag, represented as bounding boxes, along with the confidence level (min number of votes) for each bounding box.
voxel_size = 0.2

localization_params = {
    'voxel_voting': {
        'viewpoint_weight': None,  # [None, 'confidence']
        'voxel_size': voxel_size,
        'scoring_method': 'normalized_votes',  # ['normalized_votes', 'votes']
    },

    'clustering': {
        'algorithm': 'dbscan',  # ['dbscan', 'hdbscan']
        'dbscan_kwargs': {
            'eps': 2 * voxel_size,
            'min_points': 5,
            'print_progress': False,
        },

        'clustering_levels': [0.0, 0.25, 0.5, 0.75],  # only used if 'scoring_method' == 'normalized_votes'
        'bounding_box_type': 'axis_aligned',  # ['axis_aligned', 'oriented']
    },
}
loc_outputs = localization_pipeline(viewpoints, localization_params, verbose=False)
voxel_center_points = loc_outputs["voxel_center_points"]
voxel_scores = loc_outputs["voxel_scores"]
level_bbxes = loc_outputs["level_bbxes"]

# Visualize localizations
# Visualize the voxel voting results. Voxel points are colored by their corresponding number of votes.
voxel_center_points_color = cm.viridis(voxel_scores / voxel_scores.max())[:, :3]

voxel_pcd = o3d.geometry.PointCloud()
voxel_pcd.points = o3d.utility.Vector3dVector(voxel_center_points)
voxel_pcd.colors = o3d.utility.Vector3dVector(voxel_center_points_color)

o3d.visualization.draw_geometries([scene_mesh, voxel_pcd])

# Visualize proposed localization bounding boxes. Bounding boxes are colored by their confidence levels corresponding to the minimum number of votes for voxels within the bounding box.
confidences = [l for l, _ in level_bbxes]
boxes = [b for _, b in level_bbxes]
max_conf = np.max(confidences)

boxes_linemeshes = []
for conf, box in zip(confidences, boxes):
    color = cm.viridis(conf / max_conf)[:3]

    boxes_linemeshes += box_to_linemesh(
        box,
        color=color,
        radius=0.02
    ).cylinder_segments

o3d.visualization.draw_geometries([scene_mesh] + boxes_linemeshes)