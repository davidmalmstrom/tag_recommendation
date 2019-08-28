from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization

def get_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', name='dense', input_dim=100))
    # model.add(BatchNormalization(name='bn'))
    model.add(Dense(1, activation='sigmoid', name='activation'))
    model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model


if __name__ == "__main__":
    model = get_model()
    # Generate dummy data
    import numpy as np
    np.random.seed(1)
    data = np.random.random((1000, 100))
    labels = np.random.randint(2, size=(1000, 1))
    print("TRAINING MODEL 1")
    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, labels, epochs=70, batch_size=32)
    model.save_weights("test_save")
