# # Imports
# import numpy as np
# import os

# from tflite_model_maker.config import ExportFormat, QuantizationConfig
# from tflite_model_maker import model_spec
# from tflite_model_maker import object_detector

# from tflite_support import metadata

# import tensorflow as tf
# assert tf.__version__.startswith('2')

import numpy as np
import os
import matplotlib.pyplot as plt
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_support import metadata
from tflite_model_maker.object_detector import DataLoader
import tensorflow as tf
import datetime
import subprocess
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Confirm TF Version
print("\nTensorflow Version:")
print(tf.__version__)
print()


# data = object_detector.DataLoader.from_pascal_voc(
#     image_dir='/content/drive/MyDrive/images/images/',
#     label_map='/content/drive/MyDrive/images/images/classes.txt'
# )
# model = object_detector.create(train_data=data, model_spec=object_detector.EfficientDetModelSpec('efficientdet-lite0'), epochs=50)
# model.evaluate(data)
# model.export(export_dir='output_directory')


##############################################################
# Load Dataset
train_data = object_detector.DataLoader.from_pascal_voc(
    '/content/drive/MyDrive/3_obj_dataset/train',
    '/content/drive/MyDrive/3_obj_dataset/train',
    ['BIG', 'Medium', 'Small']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    '/content/drive/MyDrive/3_obj_dataset/test',
    '/content/drive/MyDrive/3_obj_dataset/test',
      ['BIG', 'Medium', 'Small']
)



spec = object_detector.EfficientDetSpec(
    model_name='efficientdet-lite2',
    uri='https://tfhub.dev/tensorflow/efficientdet/lite2/feature-vector/1',
    model_dir='/content/checkpoints',
    hparams={'max_instances_per_image': 8000}
)


# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# command = f"tensorboard --logdir={log_dir}"

# subprocess.run(command, shell=True)
# Train the model
model = object_detector.create(
    train_data=train_data,
    model_spec=spec,
    batch_size=4,
    train_whole_model=True,
    epochs=20,
    validation_data=val_data
)

##################################### save model and print option

# Export the model to TFLite format
model.export(export_dir='/content/drive/MyDrive/', tflite_filename='object_detect_model.tflite')

# Load the TFLite model for evaluation
tflite_model_path = '/content/drive/MyDrive/object_detect_model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Evaluate the model
eval_result = model.evaluate(val_data)

# Print COCO metrics for the original model
print("COCO metrics:")
for label, metric_value in eval_result.items():
    print(f"{label}: {metric_value}")

# Evaluate the TFLite model
tflite_eval_result = model.evaluate_tflite(tflite_model_path, val_data)

# Print COCO metrics for the TFLite model
print("COCO metrics tflite")
for label, metric_value in tflite_eval_result.items():
    print(f"{label}: {metric_value}")






############################################################ no need for now
# Visualize training accuracy and validation accuracy
# training_accuracy = model.history.history['accuracy']
# validation_accuracy = model.history.history['val_accuracy']

# epochs = range(1, len(training_accuracy) + 1)

# plt.plot(epochs, training_accuracy, 'bo', label='Training Accuracy')
# plt.plot(epochs, validation_accuracy, 'r', label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # Visualize COCO metrics for the TFLite model
# coco_metrics_tflite = tflite_eval_result['COCO mAP']
# coco_metric_labels = list(coco_metrics_tflite.keys())
# coco_metric_values = list(coco_metrics_tflite.values())

# plt.bar(coco_metric_labels, coco_metric_values)
# plt.title('COCO Metrics for TFLite Model')
# plt.xlabel('Metric')
# plt.ylabel('Value')
# plt.show()


# # Load model spec
# spec = object_detector.EfficientDetSpec(
#   model_name='efficientdet-lite2',
#   uri='https://tfhub.dev/tensorflow/efficientdet/lite2/feature-vector/1',
#   model_dir='/content/checkpoints',
#   hparams={'max_instances_per_image': 8000})

# # Train the model
# model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)


# # Evaluate the model
# eval_result = model.evaluate(val_data)

# # Print COCO metrics
# print("COCO metrics:")
# for label, metric_value in eval_result.items():
#     print(f"{label}: {metric_value}")

# # Add a line break after all the items have been printed
# print()

# # Export the model
# model.export(export_dir='.', tflite_filename='obj.tflite')

# # Evaluate the tflite model
# tflite_eval_result = model.evaluate_tflite('obj.tflite', val_data)

# # Print COCO metrics for tflite
# print("COCO metrics tflite")
# for label, metric_value in tflite_eval_result.items():
#     print(f"{label}: {metric_value}")

