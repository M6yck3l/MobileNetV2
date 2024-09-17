from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('lcc-train04b-weight_all/mobilenetv2-epoch_15.keras')
classes = ['Live', 'Spoof']

def preprocess_image(image):
    # Redimensionar la imagen a 224x224 (el tamaño que espera MobileNetV2)
    image_resized = cv2.resize(image, (224, 224))
    
    # Normalizar la imagen
    image_normalized = image_resized.astype('float32') / 255.0
    
    # Expandir las dimensiones para que sea compatible con el modelo (debe tener la forma [1, 224, 224, 3])
    image_expanded = np.expand_dims(image_normalized, axis=0)
    
    return image_expanded

def classify_face(image, model):
    image_preprocessed = preprocess_image(image)
    prediction = model.predict(image_preprocessed)
    predicted_class = 1 if prediction[0][0] >= 0.3 else 0
    print("Pred:", prediction, predicted_class)
    
    return classes[predicted_class]

def run_facial_spoofing_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Ordenar los rostros por tamaño (ancho * altura) en orden descendente
            faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)
            
            # Seleccionar el rostro más grande (el más cercano)
            (x, y, w, h) = faces[0]
            face = frame[y:y+h, x:x+w]
            
            label = classify_face(face, model)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        cv2.imshow('Facial Spoofing Detection', frame)
        
        # Salir del bucle presionando 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_facial_spoofing_detection()