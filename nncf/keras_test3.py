from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Layer, Concatenate, Input
from keras.initializers import Constant
from keras_test import get_model
from keras import backend as K
from keras.constraints import Constraint
from keras import optimizers
from keras.callbacks import LambdaCallback
import numpy as np
from keras.layers.normalization import BatchNormalization

class ScaledLayer(Layer):
    def __init__(self, **kwargs):
        super(ScaledLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        init = Constant(value=0.4)
        self.W = self.add_weight(shape=(1,), # Create a trainable weight variable for this layer.
                                 initializer=init, trainable=True, name='scaling_factor',
                                 constraint=Between(0.0, 1))
    
        super(ScaledLayer, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x, mask=None):
        # res = K.tf.add(x, K.tf.Variable(-1.0))
        # return K.tf.add(res, K.tf.multiply(self.W, 2))

        # x, auxInput = inpsut

        batch_size = K.shape(x)[0]
        constant = K.variable(np.ones((1, 32)))
        constant = K.tile(constant, (batch_size, 1))
        # ones = K.ones_like(K.placeholder((None, 32)))
        
        return Concatenate()([K.tf.multiply(constant, self.W), x])
        # return K.tf.multiply(x, self.W)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim*2)


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
def print_function(*args):
    print("                         " + str(model.layers[3].get_weights()[0][0]))
    print("                         " + str(np.mean(np.abs(model.layers[1].get_weights()[0]))))
print_weights = LambdaCallback(on_epoch_end=print_function)

# const_vector = K.variable(np.ones((1,32)))
# const_vector = K.repeat_elements(const_vector, rep=model.batch_input_shape)

inputs = Input(shape=(100,))

n = K.variable(np.ones(32))
auxInput = Input(tensor=n)

x = Dense(32, activation='relu', name='dense', trainable=False)(inputs)
x = BatchNormalization(name='bn', trainable=True)(x)
x = ScaledLayer()(x)
predictions = Dense(1, activation='sigmoid', name='activation', trainable=False)(x)

model = Model(inputs=inputs, outputs=predictions)
adam = optimizers.Adam()
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])

dense_weights = trained_model.get_layer('dense').get_weights()
# bn_weights = trained_model.get_layer('bn').get_weights()
activation_weights = trained_model.get_layer('activation').get_weights()

new_weights = np.concatenate((np.ones((32, 1)), activation_weights[0]), axis=0)
model.get_layer('dense').set_weights(dense_weights)
# model.get_layer('bn').set_weights(bn_weights)
model.get_layer('activation').set_weights((new_weights, activation_weights[1]))


# Generate dummy data
import numpy as np
np.random.seed(1)
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

print(model.summary())
print(model.layers[3].get_weights()[0][0])
# Train the model, iterating on the data in batches of 32 samples
hist = model.fit(data, labels, epochs=70, batch_size=32, callbacks=[print_weights])
print("asd")