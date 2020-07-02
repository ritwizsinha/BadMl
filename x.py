import cv2
import numpy as np
from string import ascii_uppercase
from keras.models import model_from_json
count = 0
map = dict()
for i in ascii_uppercase:
    if( i != 'J' and i !='Z'):
        map[count] = i
        count+=1

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]
def keras_predict(model, image):
    data = np.asarray( image, dtype="int32" )
    
    pred_probab = model.predict(data)[0]
# softmax gives probability for all the alphabets hence we have to choose the maximum probability alphabet 
    pred_class = list(pred_probab).index(max(pred_probab))
    return pred_class

def main(): 
# capturing the image from webcam 
    cam_capture = cv2.VideoCapture(0)
    while True: 
        _, image_frame = cam_capture.read()
        image_frame = cv2.flip(image_frame,1)
        cv2.rectangle(image_frame,(200,200),(400,400),1)
        cv2.imshow("frmae",image_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# to crop required part
        im2 = image_frame[200:400,200:400]
# convert to grayscale 
     
        image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
# blurring the image 
  
        image_grayscale_blurred =cv2.GaussianBlur(image_grayscale, (15,15), 0)
# resize the image to 28x28
        im3 = cv2.resize(image_grayscale_blurred, (28,28), interpolation = cv2.INTER_AREA)

# expand the dimensions from 28x28 to 1x28x28x1
        im4 = np.resize(im3, (28, 28, 1))
        im5 = np.expand_dims(im4, axis=0)
        print(map[keras_predict(loaded_model,im5)])
    cv2.destroyAllWindows()
main()
