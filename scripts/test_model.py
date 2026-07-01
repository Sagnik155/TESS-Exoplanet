from models.cnn_bilstm_attention import build_model

model = build_model()

model.summary()

import tensorflow as tf

dummy = tf.random.normal((4, 2048, 1))

prediction = model(dummy)

print(prediction.shape)