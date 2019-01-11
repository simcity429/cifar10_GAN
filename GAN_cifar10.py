from keras.datasets import cifar10
import numpy as np
from PIL import Image
import os
import keras.backend as K
from keras import models, layers, optimizers
K.set_image_data_format('channels_first')




class GAN(models.Sequential):
    def __init__(self, input_dim=64):
        super().__init__()
        self.input_dim = input_dim

        self.generator = self.GENERATOR()
        self.discriminator = self.DISCRIMINATOR()
        self.add(self.generator)
        self.discriminator.trainable = False
        self.add(self.discriminator)

        self.compile_all()

    def compile_all(self):
        d_optim = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999)
        g_optim = optimizers.Adam(lr=0.0006, beta_1=0.9, beta_2=0.999)
        self.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    def GENERATOR(self):
        input_dim = self.input_dim
        model = models.Sequential()
        model.add(layers.Dense(1024, activation='tanh', input_dim=input_dim))
        model.add(layers.Dense(128*8*8, activation='tanh'))
        model.add(layers.BatchNormalization())
        model.add(layers.Reshape((128, 8, 8), input_shape=(128*8*8, )))
        model.add(layers.UpSampling2D(size=(2,2)))
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh'))
        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(3, (5, 5), padding='same', activation='tanh'))
        return model

    def DISCRIMINATOR(self):
        model = models.Sequential()
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh', input_shape=(3, 32, 32)))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(128, (5, 5), activation='tanh'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='tanh'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def get_z(self, In):
        input_dim = self.input_dim
        return np.random.uniform(-1, 1, (In, input_dim))

    def train_both(self, x):
        In = x.shape[0]
        z = self.get_z(In)
        w = self.generator.predict(z, verbose=0)
        xw = np.concatenate((x, w))
        y2 = np.array([1] * In + [0] * In)
        d_loss = self.discriminator.train_on_batch(xw, y2)
        z = self.get_z(In)
        self.discriminator.trainable = False
        for i in range(2):
            g_loss = self.train_on_batch(z, np.array([1]*In))
        self.discriminator.trainable = True

        return d_loss, g_loss


def save_image(generated_images, output_fold, epoch, index):
    num = generated_images.shape[0]
    for i in range(int(num/5)):
        image = generated_images[i]
        image = image*127.5 + 127.5
        image = image.transpose(1, 2, 0)
        Image.fromarray(image.astype(np.uint8)).save(output_fold + '/' + str(epoch) + '_' + str(index) + '_' +  str(i) + '.png')

def load_data(n_train):
    (X_train, y_train), (_, _) = cifar10.load_data()
    return X_train[:n_train]

def get_x(X_train, index, BATCH_SIZE):
    return X_train[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]

def train():
    BATCH_SIZE = 100
    epochs  = 5000
    output_fold = 'gan_generated'
    input_dim = 64
    n_train = 10000
    os.makedirs(output_fold, exist_ok=True)
    X_train = load_data(n_train)
    np.random.shuffle(X_train)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    gan = GAN(input_dim)
    d_loss_ll = []
    g_loss_ll = []
    for epoch in range(epochs):
        print("epoch:", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        d_loss_l = []
        g_loss_l = []

        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            print('index', index)
            x = get_x(X_train, index, BATCH_SIZE)

            d_loss, g_loss = gan.train_both(x)

            d_loss_l.append(d_loss)
            g_loss_l.append(g_loss)
            print('d_loss, g_loss', d_loss, g_loss)
        if epoch % 10 == 1:
            z = gan.get_z(x.shape[0])
            w = gan.generator.predict(z, verbose=0)
            save_image(w, output_fold, epoch, '_')
            d_loss_ll.append(d_loss_l)
            g_loss_ll.append(g_loss_l)


if __name__ == '__main__':
    train()


