from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the trained model
model = load_model('soil_cnn_model.h5')  # Adjust the filename if needed

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy')  # Adjust the filename if needed

# Load and preprocess the input image
input_image_path = 'C:/Users/ADMIN/Desktop/Soil_Water_Analysis_Application/test.jpg'
input_image = load_img(input_image_path, target_size=(150, 150))
input_image_array = img_to_array(input_image)
input_image_array = input_image_array.reshape((1,) + input_image_array.shape)
input_image_array /= 255.0  # Rescale to the range [0, 1]

# Make predictions
predictions = model.predict(input_image_array)

# Decode predictions
decoded_predictions = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
print('Predicted Soil Type:', decoded_predictions[0])
