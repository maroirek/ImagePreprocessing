import cv2 as cv
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import skimage.measure
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans as KM
import scipy.cluster.hierarchy as sch
from matplotlib.figure import Figure
from conv import convolved_2d
import skimage.feature
import skimage.viewer

def filters(path):  
    img0=cv.imread(path)
    img1 = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
    img2 = cv.blur(img1, (3, 3))
    img3 = cv.blur(img1, (25, 25))
    img4= cv.medianBlur(img1,9)
    img5 =  cv.GaussianBlur(img1,(25,25),0)
    im= Image.open(path)
    img6 = im.filter(ImageFilter.MaxFilter)
    img7= cv.bilateralFilter(img1,9,75,75)
    
    title=['Original','Averaging - 3x3','Averaging - 25x25',' median 25',' gaussian','max','bilateral']
    plt.figure().set_size_inches(25,25)
    
    for i,j,title in zip(range(8),range(1,8), title): 
        plt.subplot(f'33{j}'),plt.imshow(f'img{i}'),plt.title(title)   
        # instead of storing all the images in an array (waste of memory) and then browse that array,
        # I named the images in a way i can browse them
    plt.imshow()    
    plt.savefig('NoiseElimination')


def segmentation(path) :  
    image=cv.imread(path)
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    grayim = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    #simple thresholding
    ret,th1 = cv.threshold(grayim,125,255,cv.THRESH_BINARY)

    #adaptive thresholding
    th2 = cv.adaptiveThreshold(grayim,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
    th3 = cv.adaptiveThreshold(grayim ,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

    # Otsu thresholding
    ret4,th4 = cv.threshold(grayim ,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Connected Component Analysis (with skimage directly)
    CCA, count = skimage.measure.label(th4,  return_num=True)

    # Watershed Segmentation
    distance = ndi.distance_transform_edt(grayim)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=grayim)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=grayim)

    # Clustering based Segmentation 
    # 1- With Kmeans
    img1 = grayim.flatten()
    img1 = grayim.reshape((-1, 1))
    img1_32 = np.float32(img1)
    k_means = KM(n_clusters=5).fit(img1) 
    Kmeans_seg = img1_32[k_means.labels_]
    Kmeans_seg = Kmeans_seg.reshape(grayim.shape)

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
    KM1001 = KM100.reshape(grayim.shape)

    final_clusters = sch.fcluster(heyhey,5,criterion='maxclust')    

    hierar=np.zeros(KM100.shape[0])
    for i in range(KM100.shape[0]):
        hierar[int(i)]=final_clusters[int(KM100[int(i)]-1)]
    
    
    hierarf= hierar.reshape(grayim.shape)
    
    title=['Original','Simple thresholding','Adaptive thresholding with Mean filter',
           'Adaptive thresholding with Gaussian filter','Otsu thresholing',
           'Connected Component Analysis',
           'Watershed Segmentation','Kmeans with 5 clusters', 'Hierarchical clustering with 5 clusters']
    
    images= [img,th1,th2,th3,th4,CCA,labels,Kmeans_seg,hierarf]
    
    plt.figure().set_size_inches(25,25)
    for i,img,title in zip(range(1,10),images, title):
        plt.subplot(f'33{i}'),plt.imshow(img),plt.title(title)
    plt.imshow()
    plt.savefig('Segmentation.png')
    
    
def edge(path):
       
    image=cv.imread(path)
    grayim = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    # Sobel
    kernelxs = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    kernelys = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobelx = cv.filter2D(grayim, -1, kernelxs)
    sobely = cv.filter2D(grayim, -1, kernelys)

    amp_sobel_carre=np.square(sobelx.astype(np.float32))+np.square(sobely.astype(np.float32))
    amp_sobel=np.sqrt(amp_sobel_carre)
    amp_sobel=np.uint8(amp_sobel)
    
    edge = cv.Canny(image=grayim,threshold1=1,threshold2=50)
    
    # Prewitt
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv.filter2D(grayim, -1, kernelx)
    img_prewitty = cv.filter2D(grayim, -1, kernely)
    prewitt= img_prewittx + img_prewitty
    
    # Laplacien 4 connexités
    lap= cv.Laplacian(grayim, ddepth=-1, ksize=1)
    
    #scharr 
    scharr_X = cv.Scharr(image, cv.CV_64F, 1, 0) 
    scharr_X_abs = np.uint8(np.absolute(scharr_X)) 
    scharr_Y = cv.Scharr(image, cv.CV_64F, 0, 1) 
    scharr_Y_abs = np.uint8(np.absolute(scharr_Y)) 
    scharr= cv.bitwise_or(scharr_Y_abs,scharr_X_abs) 
    
    title=['Sobel','Canny','Prewitt','Laplacien','Scharr']
    images=[amp_sobel,edge,prewitt,lap,scharr]

    plt.figure().set_size_inches(25,25)
    for i,img, title in zip(range(1,6), images, title) :
        plt.subplot(f'23{i}'),plt.imshow(img,cmap=plt.cm.gray),plt.title()
    plt.imshow()
    plt.savefig('EdgeDetection.png')
    
# segmentation("image/1.jpg")
filters("image/1.jpg")
# edge("image/1.jpg")