from pathlib import Path
from environs import Env

base_path = Path(__file__).absolute().parent
static_path = base_path / 'static'
faces_path = static_path / 'faces'
dc_path = base_path / 'cache' / 'dc'

env = Env()
env.read_env()  # read .env file, if it exists  str(base_path)


camera_index = env.int('CV2_CAMERA_INDEX', default=0)
add_capture_timer = int(3)  # seconds

use_cam_ip: bool = env.bool('USE_CAMP_IP', default=False)
cam_ip_url: str = env.str('CAM_IP_URL', default='')

if __name__ == '__main__':
    print(base_path)
