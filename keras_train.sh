#!/usr/bin/env bash

export DATA_DIR="/media/trongpq/HDD/ai_data/driver-ocr/tf_record/201904111129_digits_vgg16"
export MODEL_NAME=lenet5
export TRAIN_OUTPUT_DIR=/media/trongpq/HDD/ai_data/driver-ocr/checkpoints/digit/`date +"%Y%m%d%H%M"`_${MODEL_NAME}
mkdir -p ${TRAIN_OUTPUT_DIR}
cp ./keras_train.sh ${TRAIN_OUTPUT_DIR}/`date +"%Y%m%d%H%M"`_keras_train.sh

python keras_train_classifier.py --dataset_dir="${DATA_DIR}" \
                    --output_dir="${TRAIN_OUTPUT_DIR}" \
                    --validation_split=0.1 \
                    --batch_size=128 \
                    --num_classes=10 \
                    --image_size=50 \
                    --epochs=200 \
                    --monitor_checkpoint="acc" \
                    --monitor_mode="max" \
                    --optimizer="adadelta" \
                    --lr=0.001 \
                    --loss="categorical_crossentropy" \
                    --using_augmentation=True \
#                    --pretrained_weights="/media/trongpq/HDD/ai_data/driver-ocr/checkpoints/digit/201904161539_lenet5/weights-199-0.96.hdf5"
