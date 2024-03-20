import pickle, cv2, cvzone, sys
import numpy as np

drawing = False
points = []

def recover_spots():
    global spots
    try:
        with open('spots.pkl', 'rb') as f:
          spots = pickle.load(f)
    except:
        spots = {}

def spots_selection_ctrl(event, x, y, flag, param):
    global drawing, points, spots
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))
        drawing = False
        approx_points = cv2.approxPolyDP(np.array([points]), 5, True); points = []
        spots.update({f"{len(spots)+1}":approx_points})
    elif event == cv2.EVENT_RBUTTONDOWN:
        remove_spot(x, y)
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        remove_all_spots()
        
def remove_spot(x, y):
    global spots
    points = list(spots.values())
    names = list(spots.keys())
    for name, pts in zip(names, points):
        print(pts.shape)
        print(pts)
        print(x, y)
        if int(cv2.pointPolygonTest(pts, (x, y), False)) == 1 \
            or np.linalg.norm(pts[0] - [x, y]) < 20:
            spots.pop(name, None)  
            rearrange()
            
def remove_all_spots():
    global spots
    spots = {}
            
def rearrange():
    global spots
    new_dict = {}
    for idx, (key, value) in enumerate(spots.items(), start=1):
        new_dict[str(idx)] = value
    spots = new_dict    
        
def draw_spots(frame):
    global spots
    points = list(spots.values())
    names = list(spots.keys())
    for name, pts in zip(names, points):
        cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
        cvzone.putTextRect(frame, f'{name}', (pts[0][0][0], pts[0][0][1]), 1, 2)
    
def store_spots():
    with open('spots.pkl', 'wb') as f:
        pickle.dump(spots, f)

def dft(data_loc, type):
    recover_spots()
    if type == 'vid':
        cap = cv2.VideoCapture(data_loc)

    while True:
        if type == 'vid':
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif type == 'img':
            frame = cv2.imread(data_loc)
    
        h, w, _ = frame.shape
        ratio = float(1200/w)
        new_w = int(ratio*w); new_h = int(ratio*h)
        frame = cv2.resize(frame, (new_w, new_h))
        draw_spots(frame)
        cv2.imshow('Frame', frame)
        cv2.setMouseCallback('Frame', spots_selection_ctrl)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            if type == 'vid':
                cap.release()
            cv2.destroyAllWindows()
            sys.exit()
        elif key == ord('s'):
            store_spots()
            print('Saved')

if __name__ == "__main__":
    dft('test/parking2.jpg', 'img')