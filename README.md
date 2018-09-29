# visualizing_convnets
This is the code repository for my Medium post **Understanding your Convolution network with Visualizations**
I have used the following dataset https://www.kaggle.com/alxmamaev/flowers-recognition for training my InceptionV3 model.

You can use **create_train_val_test.py** to create Training, Validation and Test directories out of original datasets.
After that you can just run **model_training_and_visualization.py** to train the model and produce all the visualizations
shown in my blog post. Note that the images might differ depending on what test image you are using and how your network
is trained.

Although I have tried to explain my code with comments, but for any issues you can just create a Issue and I'll try to
resolve it. I have taken the code inspiration from the book **Deep Learning in Python** by **Francois Chollet**. I highly
recommend for everybody to read this book espeicially if you are new to the field.

Todo : Adding the outputs from tensorboard and explaining them in a separate blog-post
