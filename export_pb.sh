#!/usr/bin/env bash
export KERAS_MODEL_PATH="/home/tupm/Projects/keras-training/h5-model/weights-improvement-40-0.55.h5"
export PB_OUTPUT_PATH="/home/tupm/Projects/pass_ocr/pb_model/weights-improvement-40-0.55.pb"

python ./export_pb.py --input_model="${KERAS_MODEL_PATH}" \
                      --output_model="${PB_OUTPUT_PATH}" \
