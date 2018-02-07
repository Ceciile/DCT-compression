#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 06 18:37:39 2015

@author: Patricia

Modified on 10th Janv 2018

@author: Qianqian
"""
#######To clear the working memory###########
def clearall():
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]
#############################################
#clearall()    
import os        
import numpy as np
from skimage.color import rgb2gray
from skimage import data, measure,io
import skimage as sk
import matplotlib.pyplot as plt
from dct2 import dct2, idct2
import PIL as pil #pour utiliser la librairie d'écriture de fichier jpeg
import math
plt.close('all')
#img = data.astronaut(
img=io.imread('../Im3Comp100.jpg')
#si image non carré comment faire bloc qui se chevauche  cas non pris en compte?
img=io.imread('../horse.bmp')#usage d'une image carré

#img_gray = rgb2gray(img)#semble inutile dans ce cas sur mon pc affiche de useless voir ci dessous
#if img.all == img_gray.all:
#    print("useless")
    
img_gray=sk.img_as_float(img_gray)#ramene les valeur de l'image entre 0(noir) et 1(blanc)
#Convert an image to double-precision (64-bit) floating point format.

#affichage de l'image initial
plt.figure(1)
plt.imshow(img_gray,cmap='gray')
N=8
M=8
window_size = (N,M)
img_size=img_gray.shape
dctblock=np.zeros(img_size)

#affichage dct pour l'image complete
dctblockinit=np.zeros(img_size)
dctblockinit=dct2(img_gray)
plt.figure(2)
plt.imshow(np.log(1.0+dctblockinit),cmap='gray')
#newim=np.zeros(img_size)
##############################################
# l'info est principalement sur les basses frequences (en haut a gauche)
#Le premier coefficent etant celui qui contient la plus grande information 
#moyenne de l'image + varations selon la taille de l'image
#
#
#############################################
#creation de la table de quantif
def Q(n,compression):
    res =np.zeros((n,n))
    for i in range (n):
        for j in range (n):
            res[i][j]=1+((1+i+j))*compression;
    return res
########################
#pipeline dct par bloc avec/sans quantif + dct inverse

def compress(img,N,quality,quantif):
#elem de correction au tableau
#
#I size(X;Y)
    X,Y=img.shape
    #print("X,Y in compress")
    #print(X,Y)
    #N #taille bloc
    for i in np.arange(0,X-(N-1),N):
        for j in np.arange(0,Y-(N-1),N):
            blockimage=img[i:i+N,j:j+N]
            if quantif == 0:
                #sans quantif:
                dctblock[i:i+(N),j:j+(N)]=dct2(blockimage)
            else:
                #avec quantif:
                dctblock[i:i+(N),j:j+(N)]=np.round(dct2(blockimage)/Q(N,quality))
##################################################
##################################################
    #newim=np.zeros(img_size)
    newim=np.zeros(img.shape)
    ### idct2 par bloc
    for i in np.arange(0,X-(N-1),N):
        for j in np.arange(0,Y-(N-1),N):
            if quantif == 0:
                #sans quantif
                newim[i:i+(N),j:j+(N)] = idct2(dctblock[i:i+(N),j:j+(N)])
            else:
                #avec quantif
                newim[i:i+(N),j:j+(N)] = idct2( dctblock[i:i+(N),j:j+(N)]*Q(N,10))
    return newim


#calcul du psnr et ssim pour plusieur -qalite
def metrique(nbpoint):
    psnr = np.zeros(nbpoint)
    ssim = np.zeros(nbpoint)
   

    for i in range (nbpoint):#pour differente -qualité plus cette valeur est eleve moins la qualite est bonne
        newim = compress(img,2,i,1)#1 POUR AVEC QUANTIF 
        print("veuiller attendre")
        psnr[i]=measure.compare_psnr(img_gray,newim,1.0)
        ssim[i]=measure.compare_ssim(img_gray,newim)
    return psnr,ssim
 
#affichage des courbes psnr et ssim    
    
nbpoint = 15#15 pour des plus jolies courbes
xaxis = np.arange(0,nbpoint,1)
psnr,ssim=metrique(nbpoint)
plt.figure(5)
plt.title("psnr")
plt.plot(xaxis,psnr)

plt.figure(6)
plt.plot(xaxis,ssim)
plt.title("ssim ")

#test differente -qualite: plus cette valeur (1,10,25) est eleve moins la qualite est bonne 
#newim=compress(img,2,1,1)
#newim=compress(img,2,10,1)
#newim=compress(img,16,25,1)#mauvais contour

newim=compress(img,2,25,1)#mauvaise zone homogene si zoom
newim2=compress(img,8,25,1)#zone homogene ok
#affichage cheval bloc moyen zone homogene ok
plt.figure(44)
plt.imshow(newim2,cmap='gray')


#########################
#Les deux indicateurs croissent lorsque la qualite de l'image se degrade
#mais dans des plages differentes:
#Lineaire et entre 0 et 1 pour ssim
#logarithmique en partant de -66 environ (db) pour le psnr
#
#######################
#affichage de la dct par bloc
#effet de bord de compress modifie dctblock 
plt.figure(3)
plt.imshow(np.log(1.0+dctblock),cmap='gray')    
#####
#newim=idct2(dctblock)
###############################################
#pourquoi taille 8 "pour raison de complexite"
#compromis  
#chaque taille apporte des artefact different
#jpeg a choisi 8*8 pour l' optimalite 

#16*16 avec compression tres forte tres mauvais contour
#2**2 4*4 mauvais pour les zones homogenes 
###############################################
"""
psnr=measure.compare_psnr(img_gray,newim,1.0)
ssim=measure.compare_ssim(img_gray,newim)
"""
#affichage cheval grosse quantif zone homogene mauvaise petit bloc
plt.figure(4)
#newim=np.ubyte(np.round(255.0*newim,0))
#remettre le type des données ici entre 0 et 255 donc uint8
plt.imshow(newim,cmap='gray')

fich=open('madct.dat','wb')
fich.write(np.reshape(newim,-1)) 
#on étend le tableau en 1D pour pouvoir enregistrer chaque octet
fich.close()

########
"""
l'image originale (pour ceci prendre une
image au format bmp au départ ou enregistrez là en binaire sans compression):
Avant zip, la taille "horse.bmp" est 263.2 KB
apres zip, la taille est 1.1 KB
le taux de compression est (263.2 - 1.1) / 263.2 = 99.5%

celui de l'image DCT quantifiée:
Avant zip, la taille jpeg est 32.2 KB
apres zip, la taille est 1.3 KB
le taux de compression est 95.9%
"""
########

#pour sauver l'image en format jpeg pour une qualité voulue
"""
monImlu=pil.Image.open("essai.jpeg")
print( "taille= ",os.path.getsize("essai.jpeg"), "en octet")
print("compression =", 1.0*img_size[0]*img_size[1]/os.path.getsize("essai.jpeg"))

"""

###############################################################################
# 2eme partie sequence video
#preparer une image 
def prepare(Img):
    Img = (rgb2gray(Img)*255)#mise des valeur entre 0. et 255.
    Img=Img[0:128,0:128]#decoupage image carre
    Img = Img.astype(np.uint8)# valeur entre 0 et 255
    Img=compress(Img,8,1,1)#dct bloc de 8 bonne qualite avec quantif
    return Img
"""
nbim = 20
seq = np.zeros(nbim)
"""
# exemple prediction I11 et I12

I11=io.imread('../sequence_video/taxi_11.bmp')
I11=prepare(I11)
        
#affichage de I11
plt.figure(11)
plt.imshow(I11,cmap='gray')

I12=io.imread('../sequence_video/taxi_12.bmp')
I12=prepare(I12)
#affichage I12
plt.figure(12)
plt.imshow(I12,cmap='gray')


#128 * 128 ok


#recherche de bloc
#point en haut a gauche des bloc

def prediction(imgA,imgB,N,sf):
    X,Y = imgA.shape
    Ipredit =np.zeros((X,Y))#just for inititalisation
    for i in np.arange(0,X-(N-1),N):
        for j in np.arange(0,Y-(N-1),N):
            #print(j)
            mini = 255*15*15
            blockimage=imgB[i:i+N,j:j+N]
            #recheche de ce bloc dans ImgA
            #fenetre
            windowXbas = max(0,i-sf)
            windowYbas = max(0,j-sf)
            windowXhaut = min(X-sf,i+sf)
            windowYhaut = min(Y-sf,j+sf)
            for ii in range(windowXbas,windowXhaut,1):
                for jj in range(windowYbas,windowYhaut,1):
                    blockimage2=imgA[ii:ii+N,jj:jj+N]
                    if ( np.sum(np.abs(blockimage2 - blockimage)) < mini):
                        mini = np.sum(np.abs(blockimage2 - blockimage))
                        candidate = blockimage2
            Ipredit[i:i+N,j:j+N]=candidate
    return Ipredit

N = 16#taille bloc
X = 128#taille image
Y = 128#taille image
sf = 15#sizefenetre 15 32 64 meilleur image sur fentre 32 (peut etre bug)
Ipredit=prediction(I11,I12,N,sf)
#affichage image predite
plt.figure(1212)
plt.imshow(Ipredit,cmap='gray') 

#sequence prediction pas sur du resultat 
I0 = io.imread('../sequence_video/taxi_00.bmp')
I0=prepare(I0)
I1 = io.imread('../sequence_video/taxi_01.bmp')
I1 = prepare(I1)
Ipredit = prediction(I0,I1,N,sf)
for i in range (2,9):
    Iav = Ipredit#prediction precedente
    Iap =io.imread('../sequence_video/taxi_0'+str(i)+'.bmp')
    Iap = prepare(Iap)
    Ipredit = prediction(Iav,Iap,N,sf)


#img = prepare(img)
#Ipredit=prediction(img,img,N,sf) #test sur une meme image

plt.figure(15)
plt.imshow(Ipredit,cmap='gray') 
#on remarque la prediction se degrade si on en fait pluieur d'afile
"""                 
#pour chauqe bloc de l'image 2
#trouver dans ses environ dans l'image 1 le bloc qui lui ressemble le plus
#EEabs(I2(i-x,j-y) - I1(i'-x,j'-y))


#manque de temps pour la seq complete
"""
