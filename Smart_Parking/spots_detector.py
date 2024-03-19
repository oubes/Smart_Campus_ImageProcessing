import cv2
import numpy as np
import pickle
from ultralytics import YOLO
import cvzone
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

model=YOLO('yolov8s.pt')

cap=cv2.VideoCapture('test/parking2.mp4')

car_boxes = []
spot_center = []
car_center = []
used_spots = []
ratio = (0, 0)

def recover_spots():
    global spots
    try:
        with open('spots.pkl', 'rb') as f:
          spots = pickle.load(f)
    except:
        spots = {}

def find_cars(frame):
    global car_boxes
    results = model.predict(frame, verbose=False)
    car_boxes = np.intp(results[0].boxes.xyxy)
    all_labels = np.array(results[0].boxes.cls)
    desired_labels = np.array([1, 2, 3, 5, 7])
    idx = np.where(np.isin(all_labels, desired_labels))[0]
    car_boxes = car_boxes[idx]
    # for box in car_boxes:
    #     cv2.rectangle(frame, (box[:2]), (box[2:]), (255,255,255), 2)

def get_centers(type):
    global spots
    points = list(spots.values())
    if type == 'spot':
        global spot_center
        for spot in points:
            try:
                M = cv2.moments(spot)
                cx = int(M['m10'] / M['m00']); cy = int(M['m01'] / M['m00'])
                spot_center.append((cx, cy))
            except:
                pass
        return spot_center
    elif type == 'car':
        global car_center, car_boxes
        for box in car_boxes:
            w, h = abs(box[0] - box[2]), abs(box[1] - box[3])
            cx = int(box[0] + w//2); cy = int(box[1] + h//2)
            car_center.append((cx, cy))
        return car_center
        
def draw_centers(frame, used_spots):
    global spot_center, car_center
    for idx, spot in enumerate(spot_center):
        if idx in used_spots:
            cv2.circle(frame, spot, 3, (0, 0, 255), 6)
        else:
            cv2.circle(frame, spot, 3, (0, 255, 0), 6)
    # for car in  car_center:
    #     cv2.circle(frame, car, 2, (255, 0, 0), 2)
    
def dist_calc():
    global spot_center, car_center, spots
    sc = np.array(spot_center)
    cc = np.array(car_center)
    sp = spots.values()
    used_spots = []
    all_spots = list(range(1, len(sc)+1))
    for idx_i, i in enumerate(cc):
        for idx_j, j in enumerate(sp):
            j = j.reshape(-1, 2)
            if int(cv2.pointPolygonTest(j, (int(i[0]), int(i[1])), False)) == 1:
                used_spots.append(idx_j)
                
    used_spots = np.unique(used_spots).tolist()
    ratio = (len(used_spots), len(all_spots))    
    return used_spots, ratio

def add_spot_count(frame, ratio):
    cvzone.putTextRect(frame, f'{ratio[0]} / {ratio[1]}', (50, 50), 2, 2)

count = 0 
if __name__ == "__main__":
    recover_spots()
    spot_center = get_centers('spot')
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame=cv2.resize(frame,(1020,500))
        if count % 5 == 0 or count == 0:
            find_cars(frame)
            car_center = get_centers('car')
            used_spots, ratio = dist_calc()
            car_center = []
        draw_centers(frame, used_spots)   
        add_spot_count(frame, ratio)
        cv2.imshow('Frame', frame)
        count+=1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            
# %%
