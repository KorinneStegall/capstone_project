import numpy as np
import cv2
from skimage import io
from PIL import Image
  

def prediction(test, model, seg_model):
  '''
  Prediction function which takes dataframe containing ImageID as Input and perform 2 types of prediction on the image.
  1. Predict if the image has a defect or not
  2. If the image has a defect, then predict the mask of the defect
  3. Returns the ImageID, Mask and has_mask as output
  test: Dataframe containing ImageID
  model: Model to predict if the image has a defect or not
  seg_model: Model to predict the mask of the defect
  Returns: ImageID, Mask and has_mask
  Example: image_id, mask, has_mask = prediction(test, model, model_seg)
  '''

  # Directory
  directory = "./"
  # Creating empty list to store the results
  mask = []
  image_id = []
  has_mask = []
  # Iterating through each image in the test data
  for i in test.image_path:

    path = directory + str(i)
    # Reading the image
    img = io.imread(path)
    # Normalizing the image
    img = img * 1./255.
    # Reshaping the image
    img = cv2.resize(img,(256,256))
    # Converting the image into array
    img = np.array(img, dtype= np.float64)
    # Reshaping the image from 256,256,3 to 1,256,256,3
    img = np.reshape(img, (1, 256, 256, 3))
    # Making prediction on the image
    is_defect = model.predict(img)
    # If tumour is not present, append the details of the image to the list
    if np.argmax(is_defect) == 0:
      image_id.append(i)
      has_mask.append(0)
      mask.append('No mask')
      continue
    # Read the image
    img = io.imread(path)
    # Creating a empty array of shape 1,256,256,1
    X = np.empty((1, 256, 256, 3))
    # Resizing the image and coverting them to array of type float64
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype= np.float64)
    # Standardizing the image
    img -= img.mean()
    img /= img.std()
    # Converting the shape of image from 256,256,3 to 1,256,256,3
    X[0,] = img
    # Make prediction
    predict = seg_model.predict(X)
    # If the sum of predicted values is equal to 0 then there is no tumour
    if predict.round().astype(int).sum() == 0:
        image_id.append(i)
        has_mask.append(0)
        mask.append('No mask')
    else:
    # If the sum of pixel values are more than 0, then there is tumour
        image_id.append(i)
        has_mask.append(1)
        mask.append(predict)

  return image_id, mask, has_mask
