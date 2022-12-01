CONFIG_NAME=$1
PATH_TO_DATASET=$2
DATASET_CONFIG_NAME=$3
PATH_TO_EXPERIMENTS_FOLDER=$4

BATCH_SIZE=2
WORKSPACE="/root/openpcdet"
cd $WORKSPACE && ./setup_pcdet.sh

## Prepare data
cd $WORKSPACE
bash prepare_data.sh \
$PATH_TO_DATASET \
$DATASET_CONFIG_NAME \

## Train
CONFIG_FILE="cfgs/kitti_models/active_learning/$CONFIG_NAME.yaml"

PATH_TO_EXPERIMENTS=$PATH_TO_EXPERIMENTS_FOLDER/$CONFIG_NAME/`date +%Y-%m-%d_%H-%M-%S`

mkdir -p $PATH_TO_EXPERIMENTS

cd $WORKSPACE/tools

python train.py \
--cfg_file $CONFIG_FILE \
--path_to_experiments $PATH_TO_EXPERIMENTS

## Predict Labels

PREDICTION_CONFIG_FILE="cfgs/kitti_models/active_learning/${CONFIG_NAME}_pred.yaml"

python test.py \
--cfg_file $PREDICTION_CONFIG_FILE \
--batch_size $BATCH_SIZE \
--path_to_experiments $PATH_TO_EXPERIMENTS \
--save_to_file

echo "TRAINING_COMPLETE" > $PATH_TO_EXPERIMENTS/TRAINING_COMPLETE