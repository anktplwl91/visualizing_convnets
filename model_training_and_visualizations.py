'''
This python file is used for doing following things:

1. Creating training and validation generator using Keras' inbuilt ImageDataGenerator object with image augmentation.
2. Training a pre-trained InceptionV3 model.
3. Displaying and saving layer activations for first 100 layers on a test image.
4. Displaying and saving layer filters for few convolution layers
5. Displaying and saving heatmaps for test images.
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.applications.inception_v3 import InceptionV3

#-------------------
#Declaring constants
#-------------------
 
TRAIN_DIR = "train"
VALID_DIR = "validation"
IMG_SIZE = (299, 299, 3)
BATCH_SIZE = 32

#-----------------------------------------------------------------------------------
#Creating train and validation data generator using Keras' ImageDataGenerator module
#-----------------------------------------------------------------------------------

train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(IMG_SIZE[0], IMG_SIZE[1]), batch_size=BATCH_SIZE, class_mode="categorical")
val_gen = val_datagen.flow_from_directory(VALID_DIR, target_size=(IMG_SIZE[0], IMG_SIZE[1]), batch_size=BATCH_SIZE, class_mode="categorical")


#------------------------------------------
#Creating InceptionV3 model and training it
#------------------------------------------

inp = Input(IMG_SIZE)
inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=inp, input_shape=IMG_SIZE, pooling='avg')
x = inception.output
x = Dense(256, activation='relu')(x)
x = Dropout(0.1)(x)
out = Dense(5, activation='softmax')(x)

complete_model = Model(inp, out)

complete_model.compile(optimizer='adam', loss='categorical_crossentropy')
print (complete_model.summary())

history = complete_model.fit_generator(train_gen, steps_per_epoch=92, epochs=10, validation_data=val_gen, verbose=1)
print ("Saving history...")
with open('inceptionv3_histobject', 'wb') as fi:
    pickle.dump(history.history, fi)

complete_model.save('inceptionv3.h5')


#---------------------------------------------------------------------------------------
#Getting outputs for intermediate convolution layers by running prediction on test image
#---------------------------------------------------------------------------------------

layer_outputs = [layer.output for layer in complete_model.layers[:50]]
test_image = 'any test image'

img = image.load_img(test_image, target_size=(IMG_SIZE[0], IMG_SIZE[1]))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

activation_model = Model(inputs=complete_model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

layer_names = ['conv2d_1', 'activation_1', 'conv2d_4', 'activation_4', 'conv2d_9', 'activation_9']
activ_list = [activations[1], activations[3], activations[11], activations[13], activations[18], activations[20]]

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activ_list):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='plasma')
    plt.savefig(layer_name+"_grid.jpg", bbox_inches='tight')


#-------------------------------------------------
#Utility function for displaying filters as images
#-------------------------------------------------

def deprocess_image(x):
    
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#---------------------------------------------------------------------------------------------------
#Utility function for generating patterns for given layer starting from empty input image and then 
#applying Stochastic Gradient Ascent for maximizing the response of particular filter in given layer
#---------------------------------------------------------------------------------------------------

def generate_pattern(layer_name, filter_index, size=150):
    
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(80):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)

#------------------------------------------------------------------------------------------
#Generating convolution layer filters for intermediate layers using above utility functions
#------------------------------------------------------------------------------------------

layer_name = 'conv2d_4'
size = 299
margin = 5
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
        
plt.figure(figsize=(20, 20))
plt.savefig(results)


#-----------------------------------------------------------------------------------
#Here, I am initializing an InceptionV3 model and making prediction on a test image.
#Following which, I am creating a activaton heatmap of the last layer of this model,
#which is a mixed layer. This heatmap is then superimposed on the original image.
#-----------------------------------------------------------------------------------

model = InceptionV3(weights='imagenet')
img_path = 'any test image'

img = image.load_img(img_path, target_size=(IMG_SIZE[0], IMG_SIZE[1]))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print ("Predicted: ", decode_predictions(preds, top=3)[0])

#985 is the class index for class 'Daisy' in Imagenet dataset on which my model is pre-trained
flower_output = model.output[:, 985]
last_conv_layer = model.get_layer('mixed10')

grads = K.gradients(flower_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

#2048 is the number of filters/channels in 'mixed10' layer
for i in range(2048):
	conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.savefig(heatmap)

#Using cv2 to superimpose the heatmap on original image to clearly illustrate activated portion of image
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('image_name.jpg', superimposed_img)
