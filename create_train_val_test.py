'''
This python file is used to distribute original dataset of flowers into Training, Validation and Testing datasets randomly. This is important for later on using Keras'
inbuilt ImageDataGenerator module, which will easily provide us with augmented images.
'''

from os import mkdir
import glob
import random
import shutil

# You can modify this fraction for size of validation dataset and test dataset 
VAL_SIZE = 0.15
TEST_SIZE = 0.2


train_images = []
val_images = []
test_images = []


classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


current_path = # current directory path of your dataset
train_path = current_path + "\\train"
val_path = current_path + "\\validation"
test_path = current_path + "\\test"


f = []
dirs = []

for c in classes:
    
    img_list = glob.glob(orig_path + "\\" + c + "\\*.jpg")
    
    val_images = random.sample(img_list, int(VAL_SIZE * len(img_list)))
    img_list = [f for f in img_list if f not in val_images]
    test_images = random.sample(img_list, int(TEST_SIZE * len(img_list)))
    train_images = [f for f in img_list if f not in test_images]
    
    mkdir(train_path + "\\" + str(c))
    mkdir(val_path + "\\" + str(c))
    mkdir(test_path + "\\" + str(c))

    for f in train_images:
        shutil.copy(f, train_path + "\\" + str(c))

    for f in val_images:
        shutil.copy(f, val_path + "\\" + str(c))

    for f in test_images:
        shutil.copy(f, test_path + "\\" + str(c))