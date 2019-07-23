import models.base_import as models
import constants
import utility as util


def train(label, image):
    model = models.alex_net()
    model.compile(loss='mse', optimizer='adam')
    model.fit(image, label, validation_split=0.2, shuffle=True, epochs=20)
    model.save(util.model_save_path('alex_net.h5'))


if __name__ == '__main__':
    image, label = util.parse_drive_log("data/driving_log.csv")
    print (image[0].shape)
    train(label, image)
