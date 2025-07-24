import os


frequency_low = 10

base_dir = os.path.abspath(os.path.dirname(__file__))
compose_file = os.path.join(base_dir, "../docker/docker-compose.yml")
merged_config_path = os.path.join(base_dir, "tmp", "config.yml")

records_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../records"))
timestamp_file = "timestamp.txt"

landing_target_reached_file = 'target_reached.txt'
obstacles_static_list = "static_obstacles.json"