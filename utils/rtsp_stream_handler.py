import cv2
import time
import numpy as np
from typing import Tuple
import subprocess
import logging

logging.basicConfig(level=logging.INFO)


def get_camera_ip(mac_add: str) -> str:
    while True:
        try:
            output = subprocess.check_output(
                f'arp -a | findstr "{mac_add}"', shell=True
            )
            ip = output.decode("utf-8").split()[0]
            logging.info(f"IP: {ip}")
            return ip
        except subprocess.CalledProcessError:
            logging.error(
                f"Unable to get IP for MAC address: {mac_add}. Retrying in 1 second..."
            )
            time.sleep(1)


def rtsp_handler(
    username: str,
    password: str,
    mac_add: str,
    channel: int = 1,
    subtype: int = 0,
    plt=False,
) -> Tuple[bool, np.ndarray]:
    ip = get_camera_ip(mac_add)

    stream_link = f"rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}&unicast=true&proto=Onvif"
    cap = cv2.VideoCapture(stream_link)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    while True:
        ret, frame = cap.read()
        if plt is True:
            window_width = int(1920 * (5 / 7))
            window_height = int(1080 * (5 / 7))
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("frame", window_width, window_height)
            if not ret:
                logging.error("Can't receive frame (stream end?). Exiting ...")
                cap.release()
                cap = cv2.VideoCapture(stream_link)
            else:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) == ord("q"):
                    break
    cap.release()
    cv2.destroyAllWindows()
    return ret, frame


rtsp_handler(
    username="admin", password="L28FA3F4", mac_add="e4-24-6c-16-d4-f0", plt=True
)
