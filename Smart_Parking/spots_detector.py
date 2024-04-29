import cv2, cvzone, sys, json
import numpy as np
from ultralytics import YOLO

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

model=YOLO('yolov8s.pt')

def init_vars():
    global car_boxes, spot_center, car_center, used_spots, ratio, spots
    car_boxes = []
    spot_center = []
    car_center = []
    used_spots = []
    spots = {}
    ratio = (0, 0)

def recover_spots():
    global spots
    try:
        with open('spots.json', 'r') as f:
            spot = json.load(f)
            for spot in spot:
                spot_name = spot['name']
                spot_poly = np.array(spot['poly']).reshape(-1, 1, 2)
                spots[spot_name] = spot_poly
    except:
        spots = {}

def find_cars(frame):
    global car_boxes
    results = model.predict(frame, verbose=False)
    car_boxes = np.intp(results[0].boxes.xyxy.cpu().numpy())
    all_labels = np.array(results[0].boxes.cls.cpu())
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
    global spot_center, car_center, spots
    sp_k = spots.keys()
    for name, spot in zip(sp_k, spot_center):
        if name in used_spots:
            cv2.circle(frame, spot, 3, (0, 0, 255), 6)
        else:
            cv2.circle(frame, spot, 3, (0, 255, 0), 6)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'{name}', (spot[0]-12, spot[1]+20), font, fontScale=0.45, color=(255, 255, 255), thickness=1)
    # for car in  car_center:
    #     cv2.circle(frame, car, 2, (255, 0, 0), 2)
    
def dist_calc():
    global spot_center, car_center, spots
    cc = np.array(car_center)
    sp_v = spots.values()
    sp_k = spots.keys()
    used_spots = []
    all_spots = list(range(1, len(sp_v)+1))
    for idx_i, i in enumerate(cc):
        for names, poly in zip(sp_k, sp_v):
            poly = poly.reshape(-1, 2)
            if int(cv2.pointPolygonTest(poly, (int(i[0]), int(i[1])), False)) == 1:
                used_spots.append(names)
                
    used_spots = np.unique(used_spots).tolist()
    print('Used spots:', used_spots)
    ratio = [len(used_spots), len(all_spots)]    
    return used_spots, ratio

def add_spot_count(frame, ratio):
    cvzone.putTextRect(frame, f'{ratio[0]} / {ratio[1]}', (50, 50), 2, 2)

def run(img_loc, width=1200):
    init_vars()
    global spot_center, car_center
    recover_spots()
    spot_center = get_centers('spot')
    img = cv2.imread(img_loc)
    
    h, w, _ = img.shape
    if w != width:
        win_ratio = float(width/w)
        new_w = int(win_ratio*w); new_h = int(win_ratio*h)
        img = cv2.resize(img, (new_w, new_h))
        
    find_cars(img)
    car_center = get_centers('car')
    used_spots, ratio = dist_calc()
    print(used_spots, ratio)
   
def dft(vid_loc):
    init_vars()
    cap=cv2.VideoCapture(vid_loc)
    global spot_center, car_center
    count = 0 
    recover_spots()
    spot_center = get_centers('spot')
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        h, w, _ = frame.shape
        win_ratio = float(1200/w)
        new_w = int(win_ratio*w); new_h = int(win_ratio*h)
        frame = cv2.resize(frame, (new_w, new_h))
        if count % 10 == 0 or count == 0:
            find_cars(frame)
            car_center = get_centers('car')
            used_spots, ratio = dist_calc()
            car_center = []
        draw_centers(frame, used_spots)   
        add_spot_count(frame, ratio)
        cv2.imshow('Frame', frame)
        count+=1
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()


if __name__ == "__main__":
    dft('test/parking1.mp4')
    # run("test/parking2.jpg")
    
    # cap = cv2.VideoCapture("test/parking2.mp4")
    # ret, frame = cap.read()
    # cv2.imwrite('test/parking2.jpg', frame)
    
    

# %%
