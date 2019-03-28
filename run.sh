#! /usr/bin/env bash
algoID=$1
dset=$2

wrk_dir=`pwd`
data_dir="${wrk_dir}/dataset"
out_dir="${wrk_dir}/output"
model_dir="${wrk_dir}/model/${dset}"
resulr_dir="${wrk_dir}/results/${dset}"

mkdir -p $model_dir
mkdir -p $resulr_dir

cd programs
python main.py $algoID $dset $wrk_dir $resulr_dir $model_dir $data_dir
cd -
: '
After running for all 4 kinds of models run this line
cd programs
python image_visualization.py $dset $model_dir $data_dir $out_dir
cd -
'