# Generates model summary
import tensorflow as tf

model = tf.keras.models.load_model(R'..\Model\the_best_model.h5')
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model.summary()
