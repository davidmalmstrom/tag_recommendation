from keras.models import Sequential
from keras.layers import Dense, Activation, Layer, Concatenate
from keras.initializers import Constant
from keras_test import get_model
from keras import backend as K
from keras.constraints import Constraint
from keras import optimizers
from keras.callbacks import LambdaCallback


class ScaledLayer(Layer): # a scaled layer
    def __init__(self, **kwargs):
        super(ScaledLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.output_dim = input_shape[1] 
        init = Constant(value=1.2)
        self.W = self.add_weight(shape=(1,), # Create a trainable weight variable for this layer.
                                 initializer=init, trainable=True, name='scaling_factor',
                                 constraint=Between(0.0, 1.5))
    
        super(ScaledLayer, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x, mask=None):
        res = K.tf.add(x, K.tf.Variable(-1.0))
        return K.tf.add(res, K.tf.multiply(self.W, 2))
        # return K.tf.multiply(x, self.W)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class ScaledLayer2(Layer):
    def __init__(self, **kwargs):
        super(ScaledLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        init = Constant(value=0.45)
        self.W = self.add_weight(shape=(1,), # Create a trainable weight variable for this layer.
                                 initializer=init, trainable=True, name='scaling_factor',
                                 constraint=Between(0.0, 1))
    
        super(ScaledLayer, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x, mask=None):
        # res = K.tf.add(x, K.tf.Variable(-1.0))
        # return K.tf.add(res, K.tf.multiply(self.W, 2))
        return K.tf.multiply(x, self.W)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class Between(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):        
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}

trained_model = get_model()
trained_model.load_weights("test_save")

print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: 
    print("                         "  + str(model.layers[1].get_weights()[0][0]))
)

model = Sequential()
model.add(Dense(32, activation='relu', name='dense', input_dim=100, trainable=False))
model.add(ScaledLayer())
model.add(Dense(1, activation='sigmoid', name='activation', trainable=False))
rmsprop = optimizers.RMSprop(lr=0.0015)
adam = optimizers.Adam()
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])

dense_weights = trained_model.get_layer('dense').get_weights()
activation_weights = trained_model.get_layer('activation').get_weights()
model.get_layer('dense').set_weights(dense_weights)
model.get_layer('activation').set_weights(activation_weights)


# Generate dummy data
import numpy as np
np.random.seed(1)
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

print(model.layers[1].get_weights()[0][0])
# Train the model, iterating on the data in batches of 32 samples
hist = model.fit(data, labels, epochs=70, batch_size=32, callbacks=[print_weights])
print("asd")