#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %%
imagen = cv2.imread("agaves.png")
# %%
plt.imshow(imagen)
# %%
img2 = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
# %%
plt.imshow(img2)
# %%
img2
# %%
pixel_values = img2.reshape((-1,3))
# %%
pixel_values = np.float32(pixel_values)
# %%
print(pixel_values.shape)
# %%
help(cv2.TERM_CRITERIA_EPS)
# %%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# %%
help(cv2.kmeans)
# %%
# K-means clustering
# K = 4
_, labels, centers = cv2.kmeans(pixel_values, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# %%
centers = np.uint8(centers)
labels = labels.flatten()
# %%
centers
# %%
labels
# %%
segmented_image = centers[labels.flatten()]
# %%
segmented_image = segmented_image.reshape(imagen.shape)
# %%
plt.imshow(segmented_image)
# %%
plt.imshow(segmented_image)
# %%
masked_img = np.copy(imagen)
# %%
masked_img = masked_img.reshape((-1,3))
# %%
cluster = 2
masked_img[labels==cluster] = [0,0,0]
# %%
masked_img = masked_img.reshape(imagen.shape)
# %%
img3 = cv2.cvtColor(masked_img,cv2.COLOR_BGR2RGB)
plt.imshow(img3)
# %%
