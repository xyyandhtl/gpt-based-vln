import habitat
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)


DATASET_ROOT = '../datasets'
HABITAT_LAB_PATH = '../../habitat-lab/'
HM3D_CONFIG_PATH = f"{HABITAT_LAB_PATH}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
MP3D_CONFIG_PATH = f"{HABITAT_LAB_PATH}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml"
R2R_CONFIG_PATH = f"{HABITAT_LAB_PATH}/habitat-lab/habitat/config/benchmark/nav/vln_r2r.yaml"

def hm3d_config(path:str=HM3D_CONFIG_PATH,stage:str='val',episodes=200):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = f"{DATASET_ROOT}/scenes"
        habitat_config.habitat.dataset.data_path = f"{DATASET_ROOT}/habitat_task/objectnav_hm3d_v2/{{split}}/{{split}}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = f"{DATASET_ROOT}/scenes/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=90,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25
    return habitat_config
    
def mp3d_config(path:str=MP3D_CONFIG_PATH,stage:str='val',episodes=200):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = f"{DATASET_ROOT}/scenes"
        habitat_config.habitat.dataset.data_path = f"{DATASET_ROOT}/habitat_task/objectnav_mp3d_v1/{{split}}/{{split}}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = f"{DATASET_ROOT}/scenes/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25
    return habitat_config

def r2r_config(path:str=R2R_CONFIG_PATH,stage:str='val_seen',episodes=200):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = f"{DATASET_ROOT}/scenes"
        habitat_config.habitat.dataset.data_path = f"{DATASET_ROOT}/habitat_task/vln_r2r_mp3d_v1/{{split}}/{{split}}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = f"{DATASET_ROOT}/scenes/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })  
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25
    return habitat_config
