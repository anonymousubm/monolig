## Arguments
EXPERIMENT_CONFIG_NAME=$1
PREDICTION_CONFIG_NAME=$2

cd "$(dirname "$0")"

./scripts/train.py \
+experiments=$EXPERIMENT_CONFIG_NAME

./scripts/predict.py \
+experiments=$PREDICTION_CONFIG_NAME \
+previous_experiment=$EXPERIMENT_CONFIG_NAME \
EVAL_ONLY=True