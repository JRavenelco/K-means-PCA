#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen
imagen = cv2.imread("agaves.png")

# Reducir el tamaño
scale_percent = 40
width = int(imagen.shape[1] * scale_percent / 100)
height = int(imagen.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(imagen, dim, interpolation=cv2.INTER_AREA)

# Convertir a RGB para visualización
img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

# Segmentación por color (HSV)
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
green_only = cv2.bitwise_and(resized, resized, mask=mask)

# Preparar para K-means
pixel_values = green_only.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Definir criterios de K-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 4  # Número de clusters (ajustable)

# Aplicar K-means
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(resized.shape)

# Convertir a escala de grises para refinamiento
gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

# Umbralización para obtener imagen binaria
_, thresh = cv2.threshold(gray_segmented, 1, 255, cv2.THRESH_BINARY)

# Operación morfológica de cierre
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Detección de contornos
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour) > 100:  # Filtrar por área mínima
        cv2.drawContours(img_rgb, [contour], -1, (0, 255, 0), 2)

# Mostrar resultados
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Imagen Original")
plt.imshow(img_rgb)
plt.subplot(1, 3, 2)
plt.title("Segmentación K-means")
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 3)
plt.title("Contornos")
plt.imshow(img_rgb)
plt.show()
# %%
