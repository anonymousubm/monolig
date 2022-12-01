## Arguments
EXPERIMENT_CONFIG_NAME=$1

cd "$(dirname "$0")"

./scripts/train.py \
+experiments=$EXPERIMENT_CONFIG_NAME