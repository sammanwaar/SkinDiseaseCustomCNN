import tensorflow as tf

model = tf.keras.models.load_model("skin_disease_custom_cnn.h5")
num_classes = model.output_shape[-1]
print(f"Number of classes: {num_classes}")