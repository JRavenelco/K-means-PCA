#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen
imagen = cv2.imread("agaves.png")

# Reducir el tamaño
scale_percent = 50
width = int(imagen.shape[1] * scale_percent / 100)
height = int(imagen.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(imagen, dim, interpolation=cv2.INTER_AREA)

# Segmentación por color (HSV)
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
green_only = cv2.bitwise_and(resized, resized, mask=mask)
gray = cv2.cvtColor(green_only, cv2.COLOR_BGR2GRAY)

# Desenfoque Gaussiano
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Detección de bordes con Canny
edges = cv2.Canny(blurred, 30, 100)

# Umbralización adaptativa
thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)

# Operación morfológica
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Segmentación con K-means
pixel_values = thresh.reshape((-1, 1))
pixel_values = np.float32(pixel_values)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 4
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(thresh.shape)

# Detección de contornos
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour) > 100:
        cv2.drawContours(resized, [contour], -1, (0, 255, 0), 2)

# Mostrar resultados
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.title("Imagen Original")
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.subplot(1, 4, 2)
plt.title("Umbralización")
plt.imshow(thresh, cmap='gray')
plt.subplot(1, 4, 3)
plt.title("Segmentación K-means")
plt.imshow(segmented_image, cmap='gray')
plt.subplot(1, 4, 4)
plt.title("Contornos")
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.show()
# %%
