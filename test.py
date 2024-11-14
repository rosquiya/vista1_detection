import torch
from ultralytics import YOLO
import cv2

model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(model_path)

image_path = 'prueba.jpg'


image = cv2.imread(image_path)
if image is None:
    print(f"Error al cargar la imagen desde la ruta: {image_path}")
    exit()


results = model(image)


annotated_image = results[0].plot()  


output_path = 'prediccion.jpg'
cv2.imwrite(output_path, annotated_image)
print(f"Imagen guardada en: {output_path}")


cv2.imshow('Imagen con predicciones', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()