import cv2
import numpy as np
from keras.models import model_from_json

dict = {0: "Angry", 1: "Disgusted", 2: "Fear", 3: "Happy",
        4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)

model.load_weights("model.keras")
print(">> Model loaded")

cap = cv2.VideoCapture(0)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (1280, 720))

    if not ret:
        break
    
    # pre pressing da imagem
    face_detector = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    n_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=3)

    # pre processing do rosto
    for (x, y, w, h) in n_faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        cropped_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(cropped_gray_frame, (48, 48)), -1), 0)

        emotion_pred = model.predict(cropped_img)
        max_index = int(np.argmax(emotion_pred))
        cv2.putText(frame, dict[max_index], (x + 5, y - 20), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
        

    cv2.imshow("Inference", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

