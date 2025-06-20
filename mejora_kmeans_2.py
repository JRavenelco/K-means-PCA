#%% [markdown]
# # Segmentación de Imagen con K-Means Clustering
#
# Este código realiza la segmentación de una imagen utilizando el algoritmo K-Means para agrupar píxeles en clústeres según sus colores.

#%% [markdown]
# ## Importaciones
# Importamos las bibliotecas necesarias para procesamiento de imágenes y visualización.

import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% [markdown]
# ## Carga de la Imagen
# Cargamos la imagen desde el archivo especificado y verificamos que se haya cargado correctamente.

imagen = cv2.imread("agaves.png")
if imagen is None:
    print("Error: No se pudo cargar la imagen. Verifique la ruta del archivo.")
    exit()
else:
    print("Imagen cargada correctamente.")

#%% [markdown]
# ## Visualización de la Imagen Original (BGR)
# Mostramos la imagen en su formato original BGR (como la carga OpenCV).

plt.imshow(imagen)
plt.title('Imagen Original (BGR)')
plt.axis('off')
plt.show()

#%% [markdown]
# ## Conversión de Color a RGB
# Convertimos la imagen de BGR a RGB para una visualización correcta con Matplotlib.

img2 = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.title('Imagen Original (RGB)')
plt.axis('off')
plt.show()

#%% [markdown]
# ## Preparación de los Datos
# Reorganizamos la imagen en una matriz de píxeles (filas = píxeles, columnas = canales RGB) y convertimos a float32.

pixel_values = img2.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
print(f"Forma de los valores de píxeles: {pixel_values.shape}")

#%% [markdown]
# ## Criterios de Parada para K-Means
# Definimos los criterios de terminación: combinación de precisión (EPS) y número máximo de iteraciones.

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

#%% [markdown]
# ## Elección Dinámica del Número de Clústeres (K)
# Permitimos al usuario especificar el número de clústeres K.

K = int(input("Ingrese el número de clústeres K (por ejemplo, 3): "))
if K <= 0:
    print("Error: K debe ser un número positivo. Usando K=3 por defecto.")
    K = 3

#%% [markdown]
# ## Aplicación de K-Means Clustering
# Ejecutamos el algoritmo K-Means para agrupar los píxeles en K clústeres.

_, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#%% [markdown]
# ## Procesamiento de Resultados
# Convertimos los centros a enteros (uint8) y aplanamos las etiquetas.

centers = np.uint8(centers)
labels = labels.flatten()

#%% [markdown]
# ## Creación de la Imagen Segmentada
# Asignamos a cada píxel el color del centro de su clúster y reorganizamos a las dimensiones originales.

segmented_image = centers[labels]
segmented_image = segmented_image.reshape(imagen.shape)

#%% [markdown]
# ## Visualización de la Imagen Segmentada
# Mostramos la imagen segmentada.

plt.imshow(segmented_image)
plt.title(f'Imagen Segmentada con K={K}')
plt.axis('off')
plt.show()

#%% [markdown]
# ## Creación de la Imagen Enmascarada
# Creamos una copia de la imagen original y enmascaramos un clúster específico (por ejemplo, cluster=2) poniéndolo en negro.

masked_img = np.copy(imagen)
masked_img = masked_img.reshape((-1, 3))
cluster = 2  # Clúster a enmascarar (puede modificarse)
masked_img[labels == cluster] = [0, 0, 0]
masked_img = masked_img.reshape(imagen.shape)

#%% [markdown]
# ## Conversión y Visualización de la Imagen Enmascarada
# Convertimos la imagen enmascarada a RGB y la mostramos.

img3 = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
plt.imshow(img3)
plt.title(f'Imagen Enmascarada (Cluster {cluster} en negro)')
plt.axis('off')
plt.show()

#%% [markdown]
# ## Guardado de Imágenes
# Guardamos las imágenes segmentada y enmascarada en disco para uso posterior.

cv2.imwrite('segmented_image.png', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('masked_image.png', cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
print("Imágenes guardadas como 'segmented_image.png' y 'masked_image.png'")

#%% [markdown]
# ## Visualización Comparativa
# Mostramos la imagen original, segmentada y enmascarada lado a lado para facilitar la comparación.

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img2)
axes[0].set_title('Original (RGB)')
axes[1].imshow(segmented_image)
axes[1].set_title(f'Segmentada (K={K})')
axes[2].imshow(img3)
axes[2].set_title(f'Enmascarada (Cluster {cluster})')
for ax in axes:
    ax.axis('off')
plt.show()
# %%
