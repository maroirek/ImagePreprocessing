import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import skimage.measure
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans as KM
import scipy.cluster.hierarchy as sch


# reading the image 
image=cv.imread('2.png')
image1 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

#simple thresholding
ret,th1 = cv.threshold(img,125,255,cv.THRESH_BINARY)

#adaptive thresholding
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

# Otsu thresholding
ret4,th4 = cv.threshold(img,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

# Connected Component Analysis (with skimage directly)
CCA, count = skimage.measure.label(th4,  return_num=True)

# Watershed Segmentation
distance = ndi.distance_transform_edt(img)
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=img)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=img)

# Clustering based Segmentation 
# 1- With Kmeans
img1 = img.flatten()
img1 = img.reshape((-1, 1))
img1_32 = np.float32(img1)
k_means = KM(n_clusters=5).fit(img1) 
Kmeans_seg = img1_32[k_means.labels_]
Kmeans_seg = Kmeans_seg.reshape(img.shape)

# 2- With Hierarchical based on Kmeans
k_means = KM(n_clusters=100).fit(img1) 
nb_clusts = k_means.n_clusters 
centroids_after_kmeans = k_means.cluster_centers_
heyhey= sch.linkage(centroids_after_kmeans, method='ward')

cluster_index= [[] for j in range(nb_clusts)]
for i in range(img1.shape[0]):
    for j in range(nb_clusts):
        if k_means.labels_[i] == j:
            cluster_index[j].append(i)
            
KM100=np.zeros(img1.shape[0])
k=1
for i in cluster_index:
    for j in i :
        KM100[j]= k
    k=k+1     
KM1001 = KM100.reshape(img.shape)

final_clusters = sch.fcluster(heyhey,5,criterion='maxclust')

hierar=np.zeros(KM100.shape[0])
for i in range(KM100.shape[0]):
    hierar[int(i)]=final_clusters[int(KM100[int(i)]-1)]
    
    
hierarf= hierar.reshape(img.shape)


plt.figure().set_size_inches(25,25)
plt.subplot(331),plt.imshow(image1,cmap=plt.cm.gray),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(332),plt.imshow(th1,cmap=plt.cm.gray),plt.title('Simple thresholding')
plt.xticks([]), plt.yticks([])
plt.subplot(333),plt.imshow(th2,cmap=plt.cm.gray),plt.title('Adaptive thresholding with Mean filter')
plt.xticks([]), plt.yticks([])
plt.subplot(334),plt.imshow(th3,cmap=plt.cm.gray),plt.title('Adaptive thresholding with Gaussian filter')
plt.xticks([]), plt.yticks([])
plt.subplot(335),plt.imshow(th4,cmap=plt.cm.gray),plt.title('Otsu thresholing')
plt.xticks([]), plt.yticks([])
plt.subplot(336),plt.imshow(CCA),plt.title('CCA')
plt.xticks([]), plt.yticks([])
plt.subplot(337),plt.imshow(labels),plt.title('Watershed Segmentation')
plt.xticks([]), plt.yticks([])
plt.subplot(338),plt.imshow(Kmeans_seg),plt.title('Kmeans with K=5')
plt.xticks([]), plt.yticks([])
plt.subplot(339),plt.imshow(hierarf),plt.title('Hierarchical clustering with 5 clusters')
plt.xticks([]), plt.yticks([])
plt.show()


