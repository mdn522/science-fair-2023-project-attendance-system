import os
from pathlib import Path
from environs import Env

base_path = Path(__file__).absolute().parent
static_path = base_path / 'static'
faces_path = static_path / 'faces'
dc_path = base_path / 'cache' / 'dc'
attendance_history_path = base_path / 'Attendance'

static_path.mkdir(parents=True, exist_ok=True)
faces_path.mkdir(parents=True, exist_ok=True)
attendance_history_path.mkdir(exist_ok=True)

env = Env()
env.read_env()  # read .env file, if it exists  str(base_path)

camera_index = env.int('CV2_CAMERA_INDEX', default=0)
add_capture_timer = int(3)  # seconds

use_cam_ip: bool = env.bool('USE_CAM_IP', default=False)
cam_ip_url: str = env.str('CAM_IP_URL', default='')

# TEST Modes
test_use_static_image = env.bool('TEST_USE_STATIC_IMAGE', default=False)
test_static_image_name = env.str('TEST_STATIC_IMAGE_NAME', default='camera.jpg')



fonts_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
pillow_config = {
    'fonts': {
        'consolas': os.path.join(fonts_dir, 'consolab.ttf'),
        'arial': os.path.join(fonts_dir, 'Arial.ttf'),
    }
}

if __name__ == '__main__':
    print(base_path)
