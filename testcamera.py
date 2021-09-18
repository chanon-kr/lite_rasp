import cv2

video = cv2.VideoCapture(0)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
save_size = 0.5
y1 = 0
y2 = 1
x1 = 0
x2 = 1

while(video.isOpened()):
    ret, frame = video.read()
    frame = frame[int(imH*y1):int(imH*y2), int(imW*x1):int(imW*x2)]
    frame = cv2.resize(frame, (int(imW*(x2-x1)*save_size), int(imH*(y2-y1)*save_size)))
    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        print('break')
        break
video.release()
cv2.destroyAllWindows()