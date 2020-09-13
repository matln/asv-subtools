#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-7-25)

pitch=false
pitch_config=${SUBTOOLS}/conf/pitch.conf
cmvn=false
use_gpu=false
nj=16 #num-jobs
outdir=features


. "${SUBTOOLS}"/parse_options.sh

if [[ $# != 3 ]]; then
  echo "[exit] Num of parameters is not equal to 3"
  echo "usage:$0 [--pitch false|true] [--pitch-config subtools/conf/pitch.conf] [--nj 20|int] <data-dir> <feature-type> <feature-config>"
  echo "[note] Base <feature-type> could be fbank/mfcc/plp/spectrogram and the option --pitch defaults false"
  exit 1
fi

data=$1
feat_type=$2
config=$3

suffix=
cuda=

[ "$use_gpu" == "true" ] && cuda=cuda

pitch_string=
if [ $pitch == "true" ];then
  suffix=pitch
  pitch_string="--pitch-config $pitch_config"
fi

case $feat_type in 
	mfcc) ;;
	fbank) ;;
	plp) ;;
	spectrogram) ;;
	*) echo "[exit] Invalid base feature type $feat_type ,just fbank/mfcc/plp" && exit 1;;
esac

name=$(echo "$data" | sed 's/\// /g' | awk '{for(i=1;i<=NF-1;i++){printf $i"-";}printf $NF}')
# ${pitch_string} can not add double quote
"${SUBTOOLS}"/kaldi/steps/make_"${feat_type}"${suffix:+_$suffix}${cuda:+_$cuda}.sh \
  ${pitch_string} \
  --"${feat_type}"-config "$config" \
  --nj $nj \
  --cmd "run.pl" \
  "$data" \
  $outdir/"${feat_type}"/"$name"/log \
  $outdir/"${feat_type}"/"$name" || exit 1
# features/mfcc/data-mfcc_23_pitch-voxceleb1o2_train_aug

"${SUBTOOLS}"/kaldi/utils/fix_data_dir.sh "$data"

echo "Make features done."

if [ $cmvn == "true" ]; then
  "${SUBTOOLS}"/kaldi/steps/compute_cmvn_stats.sh "$data" $outdir/cmvn/log $outdir/cmvn || exit 1
  echo "Compute cmvn stats done."
fi
