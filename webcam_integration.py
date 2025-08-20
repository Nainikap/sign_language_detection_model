import cv2
import gdown
from tensorflow.keras.models import load_model
import numpy as np

file_id = "1g9LWQZ4HvaBaGbvuEATNLvz5d2UVnt4M"
gdown.download(f"https://drive.google.com/uc?id={file_id}", "sign_language_model.h5", quiet=False)
model = load_model("sign_language_model.h5")
class_names = ['1','2','3','4','5','6','7','8','9','A','B','C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V','W', 'X','Y', 'Z']


def preprocess_frame(frame):
    roi = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(frame, (64, 64))
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)  
    return roi


cap = cv2.VideoCapture(0)  
while True:

    ret, frame = cap.read() 

    if not ret:
        break  

    frame = cv2.flip(frame, 1)


    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

    roi = frame[100:300, 100:300] 
    input_frame = preprocess_frame(roi)

    pred = model.predict(input_frame)
    class_id = np.argmax(pred)
    label = class_names[class_id]


    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    cv2.putText(frame, f"Prediction: {label}", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 250, 0), 2)

    cv2.imshow("Webcam", frame)  
    cv2.imshow("Preprocessed ROI", cv2.resize(input_frame[0], (200, 200))) 


    predic = model.predict(input_frame)[0]
    print("Probabilities:", predic)
    confidence = np.max(predic)
    class_ids = np.argmax(predic)
    labels = class_names[class_ids]

    if confidence > 0.8:
        display_label = f"{labels} ({confidence:.2f})"
    else:
        display_label = "Uncertain"


cap.release()  
cv2.destroyAllWindows()  


