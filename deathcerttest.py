import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

#custom class for declare region of interest.
class ImageConstantROI():
    class CCCD(object):
        ROIS = {
            "Registration Area": [(111.4, 356.9, 200, 40)],
            "Registration Centre": [(613.7, 357.5, 500, 30)],
            "Full name": [(317.2, 417.9, 600, 50)],
            "id deceased": [(288.4,472.3,180,35)],
            "age": [(527,471.71,102,32)],
            "gender":[(740.9,471.3,400,35)],
            "documents1": [(107.6,587.6,500,40)],
            "date and time of death":[(613.0,585.3,700,50)],
            "race":[(347.87,627.98,200,60)],
            "last address":[(106.1,731.4,510,90)],
            "place of death":[(614.0,697,700,150)],
            "cause of death":[(313.5,873.6,500,60)],
            "name certifier":[(299.7,935.7,300,60)],
            "documents2":[(605.4,988.8,500,60)],
            "id certifier":[(310.9,994.6,300,40)],
            "name informant":[(258.1,1071.6,300,60)],
            "id informant":[(333.9,1125.6,200,50)],
            "documents3":[(603.8,1125.6,700,60)],
            "date of registration":[(296.0,1195.8,300,60)],
        }

#Custom function to show open cv image
def display_img(cvImg):
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,6))
    plt.imshow(cvImg)
    plt.axis('off')
    plt.show()
    
#Create a custom function to cropped image base on religion of interest
def cropImageRoi(image, roi):
    roi_cropped = image[
        int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
    ]
    return roi_cropped

#preprocessing the image
def preprocessing_image(img):
    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.multiply(gray, 1.5)
    
    #Threshold image
    th, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    return threshed
    
# Function to extract data from ID card.
def extractDataFromIdCard(img):
    for key, roi in ImageConstantROI.CCCD.ROIS.items():
        data = ''
        for r in roi:
            crop_img = cropImageRoi(img, r)
            #Preprocess the cropped image if needed
            crop_img = preprocessing_image(crop_img)
            display_img(crop_img)
            # Extract data from the image using pytesseract
            data += pytesseract.image_to_string(crop_img) + ' '
        print(f"{key} : {data.strip()}")

#SIFT TESTING HEREE#
#train image: C:Users\username\downloads\traindeathcert.png#
img2 = cv2.imread(r'YOUR TRAIN IMAGE HERE')  # trainImage

display_img(img2)
extractDataFromIdCard(img2)
