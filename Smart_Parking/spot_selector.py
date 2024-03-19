import cv2, cvzone, pickle
import numpy as np

cap = cv2.VideoCapture('test/parking2.mp4')

drawing = False
points = []

try:
    with open('spots.pkl', 'rb') as f:
        data = pickle.load(f)
        polylines, spots = data['polylines'], data['spots']
        print(polylines, spots)
except:
    polylines = []
    spots = []

def draw(event, x, y, flags, param):
    global points, drawing
    drawing = True
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        polylines.append(np.array(points, dtype='int32'))

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame = cv2.resize(frame, (1020, 500))
    for idx, polyline in enumerate(polylines):
        cv2.polylines(frame, [polyline], True, (0,0,255), 2)
        cvzone.putTextRect(frame, f'{idx+1}', tuple(polyline[0]), 1, 1)
    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', draw)
    key = cv2.waitKey(100) & 0xFF
    if key == ord('s'):
        with open('spots.pkl', 'wb') as f:
            data = {'polylines': polylines, 'spots': spots}
            pickle.dump(data, f)
cap.release()
cv2.destroyAllWindows()