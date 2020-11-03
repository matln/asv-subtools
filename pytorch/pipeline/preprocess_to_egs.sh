#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2020-02-06)


set -e

stage=0
endstage=2

force_clear=false
features_exp=features/nosil

# Do vad and traditional cmn process
nj=20
cmn=true 
compress=false # Could be false to make use of kaldi_io I/O. If true, save space of disk but increase training time.

# Remove utts
min_chunk=200
limit_utts=8

# Get chunk egs
valid_sample=true
valid_num_utts=1024
valid_split_type="--default" #"--total-spk"
sample_type="speaker_balance" # sequential | speaker_balance
chunk_num=-1
scale=1.5
overlap=0.1
valid_sample_type="every_utt" # With sampleSplit type [--total-spk] and sample type [every_utt], we will get enough spkers as more
                              # as possible and finally we get valid_num_utts * valid_chunk_num = 1024 * 2 = 2048 valid chunks.
valid_chunk_num=2
seed=1024

expected_files="utt2spk,spk2utt,feats.scp,utt2num_frames"

# . subtools/path.sh
. "${SUBTOOLS}"/parse_options.sh

if [[ $# != 2 ]]; then
  echo "[exit] Num of parameters is not equal to 2"
  echo "usage:$0 <data-dir> <egs-dir>"
  exit 1
fi

# Key params
traindata=$1
name=$(echo "$traindata" | sed 's/\// /g' | awk '{for(i=1;i<=NF-1;i++){printf $i"-";}printf $NF}')
egsdir=$2

[ ! -d "$traindata" ] && echo "The traindata [$traindata] is not exist." && exit 1

if [[ $stage -le 0 && 0 -le $endstage ]]; then
  echo "$0: stage 0"
  if [ "${force_clear}" == "true" ]; then
    rm -rf "${traindata}"_nosil
    rm -rf ${features_exp}/"${name}"_nosil
  fi

  if [ ! -d "${features_exp}" ]; then
    mkdir -p ${features_exp}
  fi

  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  if [ ! -d "${traindata}_nosil" ]; then
    "${SUBTOOLS}"/kaldi/sid/nnet3/xvector/prepare_feats_for_egs.sh \
      --nj $nj \
      --cmd "run.pl" \
      --compress $compress \
      --cmn $cmn \
      "$traindata" "${traindata}"_nosil $features_exp/"${name}"_nosil || exit 1
  else
    echo "Note, the ${traindata}_nosil is exist but force_clear is not true, so do not prepare feats again."
  fi
fi


if [[ $stage -le 1 && 1 -le $endstage ]]; then
  echo "$0: stage 1"
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 2s (200 frames) per utterance.
  # AND we also want several utterances per speaker. we'll throw out speakers
  # with fewer than 8 utterances.
  ${SUBTOOLS}/removeUtt.sh --limit-utts $limit_utts ${traindata}_nosil $min_chunk || exit 1
fi


if [[ $stage -le 2 && 2 -le $endstage ]]; then
  echo "$0: stage 2"
  [ "$egsdir" == "" ] && echo "The egsdir is not specified." && exit 1

  # valid: validation
  python3 "${SUBTOOLS}"/pytorch/pipeline/onestep/get_chunk_egs.py \
    --chunk-size=$min_chunk \
    --valid-sample=$valid_sample \
    --valid-num-utts=$valid_num_utts \
    --valid-split-type=$valid_split_type \
    --sample-type=$sample_type \
    --chunk-num=$chunk_num \
    --scale=$scale \
    --overlap=$overlap \
    --valid-chunk-num=$valid_chunk_num \
    --valid-sample-type=$valid_sample_type \
    --seed=$seed \
    --expected-files=$expected_files \
    ${traindata}_nosil $egsdir || exit 1
fi

exit 0
