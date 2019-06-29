import cv2

video_path = '/home/trongpq/Documents/AIOC/videos/VID_20190419_151901.mp4'

video_cap = cv2.VideoCapture(video_path)

while video_cap.isOpened():
    success, frame = video_cap.read()
    cv2.imshow('Video', frame)
    cv2.waitKey(25)
video_cap.release()
cv2.destroyAllWindows()