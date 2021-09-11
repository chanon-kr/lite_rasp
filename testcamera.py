import cv2

video = cv2.VideoCapture(0)

while(video.isOpened()):
    ret, frame = video.read()
    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        print('break')
        break
video.release()
cv2.destroyAllWindows()