import cv2
import time
import numpy as np
from typing import Tuple


def rtsp_handler(
    username: str, password: str, ip: str, channel: int = 1, subtype: int = 0, plt=False
) -> Tuple[bool, np.ndarray | None]:
    try:
        stream_link = f"rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}&unicast=true&proto=Onvif"
        cap = cv2.VideoCapture(stream_link)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        ret, frame = cap.read()
        if plt is True:
            window_width = int(1920 * (5 / 7))
            window_height = int(1080 * (5 / 7))
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("frame", window_width, window_height)
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
            else:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
        return ret, frame
    except Exception as e:
        print(f"An error occurred: {e}")
        return False, None


# Start Testing
def rtsp_sampling(img_every_N: int):
    """
    This function samples the RTSP stream every img_every_N seconds.

    Args:
    img_every_N (int): The number of seconds to wait between each sample.
    """
    while True:
        ret, frame = rtsp_handler(
            username="admin", password="L28FA3F4", ip="192.168.63.78", plt=True
        )
        time.sleep(img_every_N)
        print(ret, frame)


rtsp_sampling(1)
# End Testing
