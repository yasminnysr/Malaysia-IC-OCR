import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

#custom class for declare region of interest.
class ImageConstantROI():
    class CCCD(object):
        ROIS = {
            "id": [(22.6, 130.6, 700, 100)],
            "name": [(18.1, 416.2, 740, 60)],
            "address1": [(13.7, 474.9, 720, 35)],
            "address2": [(13.7,502.9,740,35)],
            "address3": [(16.4,527.4,620,35)],
            "postcode/city":[(17.2,547.4,620,35)],
            "state": [(18.1,578.1,750,60)],
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
    
#function to extract data from ID
def extractDataFromIdCard(img):
    for key, roi in ImageConstantROI.CCCD.ROIS.items():
        data = ''
        for r in roi:
            crop_img = cropImageRoi(img, r)
            crop_img = preprocessing_image(crop_img)
            display_img(crop_img)
            #Extract data from image using pytesseract
            data += pytesseract.image_to_string(crop_img) + ' '      
        print(f"{key} : {data.strip()}")
    
#SIFT TESTING HEREE#
#Example: C:\Users\username\download\ic.jpg#
#train image: identitycardtrain.jpg#
img1 = cv2.imread(r'YOUR QUERY IMAGE LOCATED HERE') # queryImage
img2 = cv2.imread(r'YOUR TRAIN IMAGE LOCATED HERE') # trainImage/template

img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1Gray, None)
kp2, des2 = sift.detectAndCompute(img2Gray, None)

# create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors.
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw good matches
matchedVis = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
display_img(matchedVis)

# Get corresponding keypoints
ptsA = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
ptsB = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Estimate homography
H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC)

# Warp image1 to align with image2
(h, w) = img2.shape[:2]
aligned = cv2.warpPerspective(img1, H, (w, h))

# Display aligned image
display_img(aligned)
extractDataFromIdCard(aligned)
