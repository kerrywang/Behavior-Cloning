import models.base_import as models
import constants
import utility as util


def train(label, image):
    model = models.inception_v3()
    print (model.summary())
    model.compile(loss='mse', optimizer='adam')
    model.fit(image, label, validation_split=0.2, shuffle=True, epochs=3)
    model.save(util.model_save_path('inception_v3.h5'))


if __name__ == '__main__':
    image, label = util.parse_drive_log("../CarND-Behavioral-Cloning-P3/data/driving_log.csv")
    print (image[0].shape)
    train(label, image)
