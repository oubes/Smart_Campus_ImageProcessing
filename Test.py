import face_recognition
import cv2

face_img1 = 'C:\\Users\\omarg\\Documents\\LV4\\GP\Project_Folder\\Auto_Attendance_V2\\output\detected_faces\\2024-01-02 23-37-59\\fr_dlib\\face_1.jpg'
face_img2 = 'C:\\Users\\omarg\\Documents\\LV4\\GP\Project_Folder\\Auto_Attendance_V2\\output\detected_faces\\2024-01-02 23-37-59\\fr_dlib\\face_1.jpg'


image1 = cv2.imread(face_img1)
image2 = cv2.imread(face_img2)

image1 = cv2.resize(image1, (224, 224))
image2 = cv2.resize(image2, (224, 224))

lface_encoded_img = face_recognition.face_encodings(image1, None, 10, 'large')
uface_encoded_img = face_recognition.face_encodings(image2, None, 10, 'large')

print(lface_encoded_img[0])

matches = face_recognition.compare_faces(lface_encoded_img, uface_encoded_img[0])
face_distances = 1 - face_recognition.face_distance(lface_encoded_img, uface_encoded_img[0])

print(matches)
print(face_distances)