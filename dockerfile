FROM tensorflow/tensorflow:2.15.0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /env

# Copy project into the image
COPY . /env/

# Minimal deps (Pillow for image loading)
RUN pip install --no-cache-dir Pillow

# Optional build args to control epochs
ARG TRAIN_EPOCHS
ARG TUNE_EPOCHS

# Train and convert to TFLite
# - Uses your dataset paths inside the image
# - Saves Keras model to /env/models/fp32_model/model.keras
# - Converts to /env/model.tflite
RUN python training/train.py \
      -i dataset/images \
      -a dataset/annotation/_annotations.coco.json \
      ${TRAIN_EPOCHS:+-e $TRAIN_EPOCHS} \
      ${TUNE_EPOCHS:+-t $TUNE_EPOCHS} && \
    python convert_model.py \
      -i /env/models/fp32_model/model.keras \
      -o /env/model.tflite