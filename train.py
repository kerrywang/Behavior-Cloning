import models.base_import as models
import constants
import utility as util


def train(label, image):
    model = models.nvidia_network()
    print (model.summary())
    model.compile(loss='mse', optimizer='adam')
    model.fit(image, label, validation_split=0.2, shuffle=True, epochs=10, batch_size=64)
    model.save(util.model_save_path('nvidia_network.h5'))


if __name__ == '__main__':
    image, label = util.parse_drive_log(["../CarND-Behavioral-Cloning-P3/data/driving_log.csv", "train_data/driving_log.csv", "backward/driving_log.csv"])
    print (image[0].shape)
    train(label, image)
