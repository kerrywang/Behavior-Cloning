from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, Lambda, Input, GlobalAveragePooling2D, Cropping2D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K

def preprocess_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    return model


def simple_mode():
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))
    return model

def alex_net():
    model = preprocess_model()
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Output Layer
    model.add(Dense(1))
    return model


def inception_v3():
    def resize(image):
        import tensorflow as tf
        return tf.image.resize_images(image, (139, 139))

#     model = preprocess_model()
    image_input = Input(shape=(160, 320, 3))
    # cropped_input = Cropping2D(cropping=((50, 20), (0, 0)))(image_input)
    resized_input = Lambda(resize)(image_input)
    normalized_input = Lambda(lambda x: x / 255.0 - 0.5)(resized_input)
    
    input_size = 139

    # Using Inception with ImageNet pre-trained weights
    inception = InceptionV3(weights='imagenet', include_top=False,input_shape=(input_size,input_size,3))
    
    inp = inception(normalized_input)
    x = GlobalAveragePooling2D()(inp)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(1)(x)
    return Model(inputs=image_input, outputs=predictions)
    
def nvidia_network():
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1))) 
    model.add(Activation('relu'))

    model.add(Flatten())    

    model.add(Dense(100))
    model.add(Activation('relu'))
    
    model.add(Dense(50))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))
    return model
    