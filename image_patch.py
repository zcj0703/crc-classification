#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import openslide
import cv2
import numpy as np
import os

 
dir_path='/mnt/nfs/tcga/'
#dir_path='C:\\Users\\ZCJ\\Pictures\\dp\\TCGA\\'
output='/mnt/nfs/tcga_512/1/'
#output='C:\\Users\\ZCJ\\Desktop\\output\\'
patchsize=512

def cut(filename):
    slide = openslide.OpenSlide(dir_path+filename)
    level_count = slide.level_count
    k=level_count-1
    [mi,ni]=slide.level_dimensions[k]
    n=slide.level_downsamples[k]

    tile = np.array(slide.read_region((0,0),k, (mi,ni)))  
    r,g,b,a = cv2.split(tile)
    img = cv2.merge([b,g,r]) 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (h,s,v) = cv2.split(hsv)
    s = cv2.GaussianBlur(s ,(15,15),0)
    th, median = cv2.threshold(s, 0, 255, cv2.THRESH_OTSU) 
    ret , thresh = cv2.threshold(s , th,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(40, 40))  
    iClose = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    iOpen= cv2.morphologyEx(iClose, cv2.MORPH_OPEN, kernel) 
    iOpen,contours0, hierarchy = cv2.findContours(iOpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contours in contours0:
        #count_con=contours0.index(contours)+1
        x,y,w,h=cv2.boundingRect(contours)
        xx=int(x*n)
        yy=int(y*n)
        ww=int(w*n)
        hh=int(h*n)
        yyy=yy
        xxx=xx
        count_x=1
        count_y=1
        while yyy<yy+hh:
            while xxx<xx+ww:
                patch = np.array(slide.read_region((xxx,yyy),0, (patchsize,patchsize)))
                r,g,b,a = cv2.split(patch)
                patchimg = cv2.merge([b,g,r])
                patchimg=cv2.cvtColor(patchimg,cv2.COLOR_BGR2GRAY)
                ret,thresh1 = cv2.threshold(patchimg,127,255,cv2.THRESH_BINARY)
                whitearea=float(np.sum(thresh1==255)/patchsize/patchsize)
                if whitearea<0.75:
                    cv2.imwrite(output+str(name_img)+'_'+str(xxx)+'_'+str(yyy)+'.jpg', patch)
                    #print(output+str(name_img)+'_'+str(xxx)+'_'+str(yyy)+'.tif')
                xxx=xxx+patchsize
                count_x=count_x+1
            yyy=yyy+patchsize
            xxx=xx
            count_y=count_y+1
            count_x=1


filenames=os.listdir(dir_path)
for filename in filenames:
    count_img=filenames.index(filename)+1
    name_img=filename.split('.')[0]
    if count_img<=1000:  
        cut(filename)
        print(filename)
 
 

    