# %%
import json, cv2
import numpy as np
img_name = 'carParkImg.png'

class spots_config:
    def __init__(self):
        self.initialization()
        
    def initialization(self):
        self.spot_list = np.zeros((0, 0), dtype='int16').reshape(0, 4, 1, 2)
        self.point_list = []
        
    def add_point(self, x, y):
        if len(self.point_list) < 4:
            current_pos = (x, y)
            self.point_list.append(current_pos)
            
    def remove_point(self, x, y):
        for idx, pos in enumerate(self.point_list):
            x1, y1 = pos
            if (abs(x-x1) < 7) and (abs(y-y1) < 7):
                self.point_list.pop(idx)
    
    def arrange_points(self, arr):
        arr_reshaped = np.array(arr).reshape(4, 2)
        x_sorted_arr = arr_reshaped[np.argsort(arr_reshaped[:, 0])]
        x1_sorted_arr = x_sorted_arr[0:2, :]
        x2_sorted_arr = x_sorted_arr[2:, :]
        y1_sorted_arr = x1_sorted_arr[np.argsort(x1_sorted_arr[:, 1])]
        y2_sorted_arr = x2_sorted_arr[np.argsort(-x2_sorted_arr[:, 1])]
        new_arr = np.append(y1_sorted_arr, y2_sorted_arr, axis=0)
        return new_arr
        
    def draw_points(self):
        Len_point_list = len(self.point_list)
        if Len_point_list <= 4:
            img = cv2.imread(img_name)
            img = self.draw_circles(img)
            if Len_point_list == 4:
                points = self.arrange_points(self.point_list)
                poly_points = np.array(points).reshape(1, 4, 1, 2)
                print(poly_points)
                self.spot_list = np.append(self.spot_list, poly_points, axis=0)
                self.point_list = []
                img = cv2.imread(img_name)
            img = self.draw_polygon(img, self.spot_list)
            cv2.imshow('img', img)
    
    def draw_circles(self, img):
        for pos in self.point_list:
            cv2.circle(img, pos, 1, (0, 0, 255), 2)
        return img
    
    def draw_polygon(self, img, spots, fill=False):
        for spot in spots:
            points = np.array(spot).reshape(-1, 1, 2)
            if fill:
                cv2.fillPoly(img, [points], (255, 255, 255))
            else:
                cv2.polylines(img, [points], True, (255, 0, 255), 2)
        return img
    
    def remove_spots(self, x, y):
        ToBeRem = []
        spot_list = self.spot_list.reshape(-1, 4, 2)
        for idx, pos in enumerate(spot_list):
            x1 = min(pos[:, 0]); x2 = max(pos[:, 0])
            y1 = min(pos[:, 1]); y2 = max(pos[:, 1])
            if x1 < x < x2 and y1 < y < y2:
                ToBeRem.append(idx)
        self.spot_list = np.delete(self.spot_list, ToBeRem, axis=0)
        
    
    
    def store_spots(self):
        with open('SpotList.json','w') as f:
            json.dump(self.spot_list.tolist(), f)
            
    def use_spots(self, img):
        with open('SpotList.json', 'r') as f:
            spots = json.load(f)
        if len(spots) >= 1:
            img = self.draw_polygon(img, spots) 
            self.spot_list = np.append(self.spot_list, np.array(spots), axis=0)
        cv2.imshow('img', img)
        
    def mask_gen(self, img):
        mask = np.zeros((img.shape))
        with open('SpotList.json', 'r') as f:
            spots = json.load(f)
        if len(spots) >= 1:
            
            mask = self.draw_polygon(mask, spots, True)
            with open('Mask.json', 'w') as f:
                json.dump(mask.tolist(), f)
        cv2.imshow('mask', mask)
        
    def mouse_click(self, events, x, y, flags, params):
        if events == cv2.EVENT_LBUTTONDOWN:
            self.add_point(x, y)
            self.draw_points()
            
        elif events == cv2.EVENT_RBUTTONDOWN:
            self.remove_point(x, y)
            self.remove_spots(x, y)
            self.draw_points()
            
        elif events == cv2.EVENT_MBUTTONDOWN:
            self.store_spots()
            cv2.destroyAllWindows()
            self.mask_gen(cv2.imread(img_name))
        
        elif events == cv2.EVENT_RBUTTONDBLCLK:
            self.initialization()
            self.draw_points()
            
    def run(self):
        img = cv2.imread(img_name)
        self.use_spots(img)
        cv2.setMouseCallback('img', self.mouse_click)
        cv2.waitKey(0); cv2.destroyAllWindows()

SP = spots_config()
SP.run()

# %%
