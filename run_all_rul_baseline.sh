CONFIG_ROOT=configs
WORKSPACE_ROOT=./workspaces_new

# Evaluate all sklearn baselines (single seed)
for folder in configs/baselines/sklearn; do
    for CONFIG in $(find $folder -type f); do
        relpath=$(echo $CONFIG | sed "s|$CONFIG_ROOT/||")
        WORKSPACE=$WORKSPACE_ROOT/${relpath%.*}
        mkdir -p $WORKSPACE
        seed=0
        python batteryml.py run $CONFIG --workspace $WORKSPACE --train --eval --skip_if_executed false --seed $seed | tee $WORKSPACE/log.$seed
    done
done

# Evaluate all nn baselines (multiple seeds)
for folder in configs/baselines/nn_models; do
    for CONFIG in $(find $folder -type f); do
        relpath=$(echo $CONFIG | sed "s|$CONFIG_ROOT/||")
        WORKSPACE=$WORKSPACE_ROOT/${relpath%.*}
        mkdir -p $WORKSPACE
        for seed in 0 1 2 3 4 5 6 7 8 9; do
            python batteryml.py run $CONFIG --workspace $WORKSPACE --train --eval --seed $seed --device cuda --skip_if_executed false | tee $WORKSPACE/log.$seed
        done
    done
done
