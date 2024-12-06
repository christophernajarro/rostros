# Importar las bibliotecas necesarias
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Diccionario de emociones
emotion_dict = {0: "Enojado", 1: "Disgusto", 2: "Miedo", 3: "Feliz",
                4: "Neutral", 5: "Triste", 6: "Sorpresa"}

# Cargar el modelo entrenado
model = load_model('modelo_emociones.keras')

# Cargar el clasificador de caras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en el frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        # Redimensionar a 48x48
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        # Realizar la predicción
        prediction = model.predict(roi_gray)
        maxindex = int(np.argmax(prediction))
        emotion = emotion_dict[maxindex]
        # Dibujar rectángulo y etiqueta
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Mostrar el frame
    cv2.imshow('Reconocimiento de Emociones', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
