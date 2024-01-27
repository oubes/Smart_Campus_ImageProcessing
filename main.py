from detection import Detect
from init import init


if __name__ == "__main__":
    init()
    from vars import detector_config
    print(detector_config)
    for model in models:
        face_locations, faces_count, taken_time = Detect(model)
