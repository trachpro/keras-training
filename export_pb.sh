#!/usr/bin/env bash
export KERAS_MODEL_PATH="/media/trongpq/HDD/ai_data/driver-ocr/checkpoints/digit/201904160859_lenet5_12/weights-181-0.99.h5"
export PB_OUTPUT_PATH="/media/trongpq/HDD/ai_data/driver-ocr/checkpoints/digit/201904160859_lenet5_12/weights-181-0.99.pb"

python ./export_pb.py --input_model="${KERAS_MODEL_PATH}" \
                      --output_model="${PB_OUTPUT_PATH}" \
