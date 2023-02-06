#!/bin/bash

set -x

cd "$(dirname "$0")"

echo "NSIZE: $1";
echo "METHOD: $2";
echo "MODEL: $3";
echo "TIME: $4";

NSIZE="$1"
METHOD="$2"
MODEL="$3"
TIME="$4"

# assert non-empty above
if [ -z "$NSIZE" ] || [ -z "$METHOD" ] || [ -z "$MODEL" ]; then
    echo "Usage: $0 NSIZE METHOD MODEL BACKEND"
    exit 1
fi


DATE="$(date +%m%d)"

# set environment variables CUDA_VISIBLE_DEVICES=""
export CUDA_VISIBLE_DEVICES=""
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

if [ $MODEL = "tensorflow" ]; then
    BACKEND="xla"
    RECORD="data/tf"
    PREFIX="tf"
elif [ $MODEL = "torch" ]; then
    BACKEND="torchjit"
    RECORD="data/torch"
    PREFIX="torch"
else
    echo "MODEL must be tensorflow or torch"
    exit 1
fi

# attempt at most 32 times
for i in {1..32}
do
    echo "Attempt $i"
    PYTHONPATH=$(pwd):$(pwd)/autoinf python dynofuzz/cli/fuzz.py fuzz.time=${TIME} fuzz.root=fuzz_output/${PREFIX}-${DATE}-${METHOD}-n${NSIZE} \
                                            mgen.record_path=${RECORD} \
                                            fuzz.save_test=fuzz_output/${PREFIX}-${DATE}-${METHOD}-n${NSIZE}.models \
                                            model.type=${MODEL} backend.type=${BACKEND} filter.type="[nan,dup,inf]" \
                                            debug.viz=true hydra.verbose=fuzz fuzz.resume=true \
                                            mgen.method=${METHOD} mgen.max_nodes=${NSIZE}
    if [ $? -eq 0 ]; then
        echo "Fuzzing succeeded after $i attempts."
        break
    else
        echo "Fuzzing crashed with exit code $?.  Respawning.." >&2
        sleep 0.5
    fi
done

echo "WAS RUNNING:"
echo "PYTHONPATH=$(pwd):$(pwd)/autoinf python dynofuzz/cli/fuzz.py fuzz.time=${TIME} fuzz.root=fuzz_output/${PREFIX}-${DATE}-${METHOD}-n${NSIZE} \\
                                            mgen.record_path=${RECORD} \\
                                            fuzz.save_test=fuzz_output/${PREFIX}-${DATE}-${METHOD}-n${NSIZE}.models \\
                                            model.type=${MODEL} backend.type=${BACKEND} filter.type="[nan,dup,inf]" \\
                                            debug.viz=true hydra.verbose=fuzz fuzz.resume=true \\
                                            mgen.method=${METHOD} mgen.max_nodes=${NSIZE}"
