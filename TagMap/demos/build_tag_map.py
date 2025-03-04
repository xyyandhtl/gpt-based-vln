import os
from tqdm import tqdm

import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
# import ipywidgets as widgets
# from IPython.display import display

from tag_mapping.datasets.matterport import (
    read_matterport_image_file,
    read_matterport_depth_file,
    read_matterport_pose_file,
    read_matterport_intrinsics_file,
    MatterportFilenameBridge
)

from tag_mapping.models import RAMTagger

from tag_mapping.filtering import (
    compute_unlikely_tags_center_crop_ensemble,
    valid_depth_frame,
)

from tag_mapping import TagMap, TagMapEntry
import uuid

scene_dir = '../datasets/demo_data'
images_dir = os.path.join(scene_dir, 'color')
depths_dir = os.path.join(scene_dir, 'depth')
poses_dir = os.path.join(scene_dir, 'poses')
mesh_path = os.path.join(scene_dir, 'mesh.ply')

# Load and visualize the scene mesh
scene_mesh = o3d.io.read_triangle_mesh(mesh_path)
o3d.visualization.draw_geometries([scene_mesh])

# Read the camera intrinsics. For simplicity, let's assume that the intrinsics of the camera are fixed across all views of the scene.
width, height, fx, fy, cx, cy, d = read_matterport_intrinsics_file(
    os.path.join(scene_dir, 'intrinsics.txt')
)

intrinsics = {
    'width': width, 'height': height,
    'fx': fx, 'fy': fy,
    'cx': cx, 'cy': cy,
}

# Build a Tag Map of the scene
# Image tagging model
# First load the image tagging model which that will generate tags for the scene's viewpoints. We use the Recognize Anything set of image tagging models.
#
# Set ram_pretrained_path to the path of the downloaded the image tagging model checkpoint.
# checkpoint.

ram_pretrained_path = '../checkpoints/ram_swin_large_14m.pth'

ram_tagger = RAMTagger(
    config={
        'ram_pretrained_path': ram_pretrained_path,
        'ram_image_size': 384,
        'vit': 'swin_l',
        'device': 'cuda',
    }
)
# Crop ensemble filtering
# We use a cropped augmented ensemble to help remove false positive tag detections by the model.
#
# First, a set of tags is generated for the unmodified image. Modified versions of the image are
# created by center cropping the unmodified image with different crop proportions.
# The modified images are also passed through the model. The final set of tags are the tags
# consistent across all images.
crop_ensemble_proportions = [0.05, 0.1]

def generate_tags(image, tagging_model):
    out = tagging_model.tag_image(image)
    tags, confidences = (out['tags'], out['confidences'])

    unlikely_tags = compute_unlikely_tags_center_crop_ensemble(
        image, tags,
        crop_ensemble_proportions,
        tagging_model
    )

    filtered_tags = [tag for tag in tags if tag not in unlikely_tags]
    filtered_tag_confidences = [conf for tag, conf in zip(tags, confidences) if tag not in unlikely_tags]

    return filtered_tags, filtered_tag_confidences

# Tag Map construction
tag_map_metadata = {
    'scene_name': os.path.basename(scene_dir),
    'intrinsics': intrinsics,
}

tag_map = TagMap(metadata=tag_map_metadata)

# Iterate over the scene viewpoints, generating tags for each pushing them to the Tag Map.
#
# Viewpoints are skipped if they are likely to be an uninformative view by checking the depth image statistics.
depth_mean_threshold = 0.6
depth_quantile_thresholds = [(0.5, 0.6)]
for image_filename in tqdm(os.listdir(images_dir)):
    filename_bridge = MatterportFilenameBridge.from_image_filename(image_filename)
    depth_filename = filename_bridge.depth_filename
    pose_filename = filename_bridge.pose_filename
    # print(f'processing {image_filename}, {depth_filename}, {pose_filename}')

    image = read_matterport_image_file(os.path.join(images_dir, image_filename))
    depth, depth_image = read_matterport_depth_file(os.path.join(depths_dir, depth_filename))
    T_cam_to_world = read_matterport_pose_file(os.path.join(poses_dir, pose_filename))

    # skip viewpoints with invalid depth
    if not valid_depth_frame(
            depth,
            mean_threshold=depth_mean_threshold,
            quantile_thresholds=depth_quantile_thresholds
    ):
        continue

    # get viewpoint tags
    tags, confidences = generate_tags(image, ram_tagger)

    # get depth statistics to store for the viewpoint
    depth_percentiles = {
        str(q): dq for q, dq in zip([0.8], np.quantile(depth, [0.8]))
    }

    # add entry corresponding to the viewpoint
    entry_uuid = uuid.uuid4()
    entry = TagMapEntry(
        pose=T_cam_to_world,
        uuid=entry_uuid,
        extras={
            'depth_percentiles': depth_percentiles,

            # Here we store the viewpoint image filenames for visualization purposes,
            # but it is not necessary to store these.
            'image_filename': image_filename,
            'depth_filename': depth_filename,
        }
    )
    tag_map.add_entry(entry)
    # print(f'added entry for {image_filename}')

    # add information on the tags of the viewpoint
    for tag, conf in zip(tags, confidences):
        tag_map.add_tag(
            tag,
            entry_uuid,

            # Here we also store the image tagging model confidence score for all viewpoint tags,
            # but currently this is information is not used downstream.
            extras={
                'confidence': conf,
            }
        )
    print(f'added {len(tags)} tags for {image_filename}')

# Inspect the Tag Map
# A tag in recognized in the scene can be selected for visualization. The tags are listed in a dropdown along with how many viewpoints that tag was recognized in.
options = []
for tag in sorted(tag_map.unique_objects):
    options.append(
        (tag + '  (' + str(len(tag_map.query(tag))) + ')', tag)
    )
print('options:', options)

# query_dropdown = widgets.Dropdown(options=options, description='Select a tag:')
# display(query_dropdown)

# Retrieve corresponding viewpoints for the selected tag.
#
# Rerun this block after changing the selection
# query_entries = tag_map.query(query_dropdown.value)
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

# Visualize the viewpoints corresponding to the tag in the scene
o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic()
o3d_intrinsics.set_intrinsics(**intrinsics)

query_frustum_geometries = []
for entry in query_entries:
    frame_frustum = o3d.geometry.LineSet.create_camera_visualization(
        intrinsic=o3d_intrinsics,
        extrinsic=np.linalg.inv(entry.pose),
    )
    frame_frustum.paint_uniform_color(np.random.rand(3))
    query_frustum_geometries.append(frame_frustum)
o3d.visualization.draw_geometries([scene_mesh] + query_frustum_geometries)