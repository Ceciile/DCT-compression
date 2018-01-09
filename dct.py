# -*- coding: utf-8 -*-
"""
Created in 2018
sudo apt-get install python-skimage
@author: qianqian
"""
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage import color, data,feature
from scipy.ndimage import convolve 
import dct
from dct2 import dct2, idct2
#import pylab
import PIL as pil #pour utiliser la librairie d'écriture de fichier jpeg

I =skio.imread("horse.bmp");
Ing = color.rgb2gray(I)

img_gray=sk.img_as_float(Ing)*255.0
plt.figure(1)
plt.rcParams['image.cmap']='gray'
plt.imshow(img_gray,cmap='gray')
#plt.show()#pylab

N=8
M=8
window_size = (N,M)
img_size=img_gray.shape
#print img_size:512
# get the largest dimension
max_dim = max(img_size)
dctblock=np.zeros(img_size)

dctblock=dct2(img_gray)

#plt.figure(3)
#plt.imshow(np.log(1.0+dctblock),cmap='gray')

newim=np.zeros(img_size)
newim=idct2(dctblock)
"""
plt.figure(4)
newim=np.ubyte(np.round(255.0*newim,0))
#remettre le type des données ici entre 0 et 255 donc uint8
plt.imshow(newim,cmap='gray')
plt.show()

DECOUPER-BLOC
QUANTIF
DE QUANTIF
ERROR

MOVEMENT-VIDEO
otil-prefe-grappty 
"""

def quantification(tab, comp):# 0~1
    t=np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            t[i,j]=np.round(tab[i,j]/(1+(1+i+j)*comp))
    return t


def DEquantification(tab, comp):
    t=np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            t[i,j]=tab[i,j]*(1+(1+i+j)*comp)
    return t

def dcoupe(img_gray,n):
    window_size = (n,n)
    img_size=img_gray.shape
    new_gray=np.zeros(img_size)
    quant=np.zeros(img_size)
    dctblock=np.zeros(window_size)
    newim=np.zeros(window_size)
    t1=np.zeros((8,8))
    t2=np.zeros((8,8))
# get the largest dimension
    max_dim = max(img_size)
    # Crop out the window and calculate
    for r in range(0,img_size[0]-n+1, n):# X-(n-1)
        for c in range(0,img_size[1]-n+1, n):
            window = img_gray[r:r+n,c:c+n]#!!!!!   i:i+n
#            dctblock=np.zeros(window_size)
            dctblock=dct2(window)

	    t1=quantification(dctblock, 5)
	    t2=DEquantification(t1, 5)
#            newim=np.zeros(window_size)
            newim=idct2(t2)#quant[r:r+n,c:c+n]
            new_gray[r:r+n,c:c+n]=newim
#    quantIm=pil.Image.fromarray(np.ubyte(np.round(255.0*quant,0)))
#pour sauver l'image en format jpeg pour une qualité voulue
#    quantIm.save('quantif.jpeg',quality=20)
    monIm=pil.Image.fromarray(np.ubyte(np.round(new_gray,0)))
#    fich=open('madct.dat','wb')
#    fich.write(np.reshape(quantIm,-1)) 
#on étend le tableau en 1D pour pouvoir enregistrer chaque octet
#    fich.close()
    return monIm
#remettre le type des données ici entre 0 et 255 donc uint8
#_____________________________________________________________  
resu8=dcoupe(img_gray,8)
plt.figure(8)
plt.imshow(resu8,cmap='gray')
"""
resu16=dcoupe(img_gray,16)
plt.figure(16)
plt.imshow(resu16,cmap='gray')

resu4=dcoupe(img_gray,4)
plt.figure(2)
plt.imshow(resu4,cmap='gray')"""

plt.show()
