import cv2
from datetime import datetime
video_capture=cv2.VideoCapture(0)
face_classifier=cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
def detect_bounding_box(vid):
    gray_image= cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)
    faces= face_classifier.detectMultiScale(gray_image,1.1,5,minSize=(40,40))
    for(x,y,w,h)in faces:
        cv2.rectangle(vid,(x,y),(x+w,y+h),(0,255,0),4)
    return faces

while True:
    result,video_frame=video_capture.read()
    if result is False:
        break

    faces=detect_bounding_box(video_frame)
    now=datetime.now()



    cv2.imshow(
        "My Face Detect Project", video_frame
    )

    if type(faces) is not tuple:
        img_name="frame_at_time"+str(now.hour)+"."+str(now.minute)+"."+str(now.second)+".png"
        cv2.imwrite(img_name,video_frame)

    k=cv2.waitKey(1)
    if k%256==27:
        print("Escape hit, closing")
        break

video_capture.release()
cv2.destroyAllWindows()
