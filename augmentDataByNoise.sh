#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2019-09-08)

# This script is used to augment data by some noise and it refers to kaldi/egs/sre16/v2/run.sh

set -e

# Default value
rirs_noises=/data1/data/RIRS_NOISES/
musan=/data1/data/musan/

reverb=true
noise=true
music=true
babble=true # a.k.a speech

sampling_rate=16000
frame_shift=0.01
# The ratio of augmented data with origin data. In this case, 4 means using all augmented data if aug-data-dir is provided.
factor=1 
# Num-jobs
nj=16
force=false

. "${SUBTOOLS}"/parse_options.sh
# . subtools/path.sh

if [[ $# != 1 && $# != 2 ]]; then
  echo "[exit] Num of parameters is not equal to 1 or 2"
  echo "usage:$0 <data-dir> [<aug-data-dir>]"
  echo "[note] if <aug-data-dir> is provided, it will contains all the data from <data-dir>"
  exit 1
fi

data=$1

[ ! -d "$data" ] && echo "Expected datadir $datadir to be exist" && exit 1

for file in utt2spk wav.scp; do
  [ ! -f $data/$file ] && echo "Expected $data/$file to exist." && exit 1
done

[ $# -eq 2 ] && aug_data_dir=$2

[[ "$reverb" != "true" && "$noise" != "true" && "$music" != "true" && "$babble" != "true" ]] && \
  echo "[exit] There should be one augmentation type form [reverb|noise|music|babble]" && exit 1

utt_num=$(cat $data/utt2spk | wc -l | awk '{print $1}')
for file in reco2dur utt2num_frames feats.scp; do
  num=$(cat $data/$file | wc -l | awk '{print $1}')
  [ $num -ne $utt_num ] && echo "[Note] The num of $data/$file is not equal to $data/utt2spk ($num/$utt_num), so mv $data/$file to $data/$file.lost." && \
                           mv -f $data/$file $data/$file.lost
done

if [ ! -f "$data"/reco2dur ]; then
	echo "...$data/reco2dur is not exist, so get it automatically..."
	if [ -f "$data"/utt2num_frames ]; then
		awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' "$data"/utt2num_frames > "$data"/reco2dur
	elif [ -f "$data"/feats.scp ]; then
		feat-to-len scp:"$data"/feats.scp ark,t:"$data"/utt2num_frames
		awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' "$data"/utt2num_frames > "$data"/reco2dur
	else
		"${SUBTOOLS}"/kaldi/utils/data/get_reco2dur.sh --nj $nj --frame-shift $frame_shift "$data"
	fi
fi

all_data=""
# data/mfcc_23_pitch
dir=$(dirname "$data")
# voxceleb1o2_train
name=$(basename "$data")

augment_dir=$dir/augment
sdata=$augment_dir/$name
additive_aug_data="$sdata"

mkdir -p "$augment_dir"

num=0

if $reverb; then
	[ ! -d $rirs_noises ] && echo "[check reverb] No such dir $rirs_noises" && exit 1

	echo "...add reverb..."
	
	if [[ ! -d ${sdata}_reverb || $force == "true" ]]; then
		rvb_opts=()
		rvb_opts+=(--rir-set-parameters "0.5, $rirs_noises/simulated_rirs/smallroom/rir_list")
		rvb_opts+=(--rir-set-parameters "0.5, $rirs_noises/simulated_rirs/mediumroom/rir_list")
	
    # Generate wav.scp，each line is the PIPE of generating reverberate data
    # Generate utt2uniq,
    # 在 utt2spk 中为<utt-id> 加上前缀，输出到${sdata}_reverb
    # 利用 utt2spk_to_spk2utt.pl 生成 spk2utt
    # 没有在 ${sdata}_reverb 中生成 reco2dur
		python3 "${SUBTOOLS}"/kaldi/steps/data/reverberate_data_dir.py \
      "${rvb_opts[@]}" \
      --speech-rvb-probability 1 \
      --pointsource-noise-addition-probability 0 \
      --isotropic-noise-addition-probability 0 \
      --num-replications 1 \
      --source-sampling-rate $sampling_rate \
      "${data}" "${sdata}"_reverb || exit 1

		# Add suffix
    "${SUBTOOLS}"/kaldi/utils/copy_data_dir.sh --utt-suffix "-reverb" "${sdata}"_reverb "${sdata}"_reverb.new
		rm -rf "${sdata}"_reverb
		mv "${sdata}"_reverb.new "${sdata}"_reverb
    top_dir=$(dirname $rirs_noises | sed 's/\//\\\//g')
    sed -i 's/ RIRS_NOISES/ '"$top_dir"'\/RIRS_NOISES/g' "${sdata}"_reverb/wav.scp
    [ -f "$data"/vad.scp ] && awk '{print $1"-reverb",$2}' "$data"/vad.scp > "${sdata}"_reverb/vad.scp
	fi

	all_data="$all_data ${sdata}_reverb"
	additive_aug_data="${additive_aug_data}"_reverb
  num=$((num + 1))
fi

musan_dir=data/musan_$sampling_rate

if $noise; then
	[ ! -d $musan/noise ] && echo "[check noise] No such dir $musan/noise" && exit 1
	
	if [ ! -d $musan_dir/musan_noise ]; then
    # 在 data/ 下产生了 musan/, musan_music/, musan_speech/, musan_noise 目录
		"${SUBTOOLS}"/kaldi/steps/data/make_musan.sh --sampling-rate $sampling_rate $musan $musan_dir || exit 1
	fi
	
	echo "...add noise..."
	
	if [[ ! -d ${sdata}_noise || $force == "true" ]];then
    # 在 ${sdata}_noise 下生成 wav.scp, ${sdata}_noise 中剩下的文件通过给 {data} 下相应文件的 <uttrance-id> 加上后缀拷贝过来
		python3 "${SUBTOOLS}"/kaldi/steps/data/augment_data_dir.py \
      --utt-suffix "noise" \
      --fg-interval 1 \
      --fg-snrs "15:10:5:0" \
      --fg-noise-dir "$musan_dir/musan_noise" "${data}" "${sdata}"_noise || exit 1
  fi
	
	all_data="$all_data ${sdata}_noise"
	additive_aug_data="${additive_aug_data}"_noise
  num=$((num+1))
fi

if $music; then
	[ ! -d $musan/music ] && echo "[check music] No such dir $musan/music" && exit 1

	if [ ! -d $musan_dir/musan_music ];then
		"${SUBTOOLS}"/kaldi/steps/data/make_musan.sh --sampling-rate $sampling_rate $musan $musan_dir || exit 1
	fi
	
	echo "...add music..."
	if [[ ! -d ${sdata}_music || $force == "true" ]];then
    # 在 ${sdata}_music 下生成 wav.scp, ${sdata}_music 中剩下的文件通过给 {data} 下相应文件的 <uttrance-id> 加上后缀拷贝过来
		python3 "${SUBTOOLS}"/kaldi/steps/data/augment_data_dir.py \
      --utt-suffix "music" \
      --bg-snrs "15:10:8:5" \
      --num-bg-noises "1" \
      --bg-noise-dir "$musan_dir/musan_music" \
      "${data}" "${sdata}"_music || exit 1
	fi

	all_data="$all_data ${sdata}_music"
	additive_aug_data="${additive_aug_data}"_music
  num=$((num+1))
fi

if $babble; then
	[ ! -d $musan/speech ] && echo "[check babble] No such dir $musan/speech" && exit 1

	if [ ! -d $musan_dir/musan_speech ];then
		"${SUBTOOLS}"/kaldi/steps/data/make_musan.sh --sampling-rate $sampling_rate $musan $musan_dir || exit 1
	fi
	
	echo "...add babble/speech..."

	if [[ ! -d ${sdata}_babble || $force == "true" ]];then
    # 在 ${sdata}_babble 下生成 wav.scp, ${sdata}_babble 中剩下的文件通过给 {data} 下相应文件的 <uttrance-id> 加上后缀拷贝过来
		python3 ${SUBTOOLS}/kaldi/steps/data/augment_data_dir.py \
      --utt-suffix "babble" \
      --bg-snrs "20:17:15:13" \
      --num-bg-noises "3:4:5:6:7" \
      --bg-noise-dir "$musan_dir/musan_speech" \
      "${data}" "${sdata}"_babble || exit 1
	fi
	
	all_data="$all_data ${sdata}_babble"
	additive_aug_data="${additive_aug_data}"_babble
  num=$((num+1))
fi

# 将 ${sdata}_reverb, ${sdata}_babble, ${sdata}_music, ${sdata}_noise 合并为 ${sdata}_reverb_noise_music_babble
if [ $num -gt 1 ];then
	echo "...combine additive aug data to $additive_aug_data..."
  # additive_aug_data: data/mfcc_23_pitch/augment/voxceleb1o2_train_reverb_noise_music_babble
	[[ ! -d $additive_aug_data || $force == "true" ]] && ${SUBTOOLS}/kaldi/utils/combine_data.sh $additive_aug_data $all_data
fi

# 借助 bc 进行数值计算
bc_path=$(command -v bc)
[ "$bc_path" == "" ] && echo -e "[exit] No bc in ($PATH)\nPlease install bc by 'yum install bc'." && exit 1

num_origin_utts=$(wc -l $data/reco2dur | awk '{print $1}')
# Use awk to replace bc to compute float value.
status=$(echo $fator $num | awk '{if($1-$2>0){print 1}else{print 0}}')
[ $status -eq 1 ] && factor=$num # Get min
num_additive_utts=$(echo $num_origin_utts $factor | awk '{print int($1*$2)}')

[ $num_additive_utts -eq 0 ] && "[exit] The factor $factor is too small" && exit 1

if [ $# -eq 2 ]; then
	subset_data=${additive_aug_data}

	if [ $factor -ne $num ];then
			echo "...get subset from $additive_aug_data to ${additive_aug_data}_$num_additive_utts..."
      # 使用类似于折半查找的方式，使得选择的子集尽可能平均。(utils/subset_scp.pl)
			[ ! -d ${additive_aug_data}_$num_additive_utts ] && \
        "${SUBTOOLS}"/kaldi/utils/subset_data_dir.sh $additive_aug_data $num_additive_utts ${additive_aug_data}_$num_additive_utts
			subset_data=${additive_aug_data}_$num_additive_utts
	fi

	echo "...generate augmented data to $aug_data_dir..."
  # aug_data_dir: data/mfcc_23_pitch/voxceleb1o2_train_aug
  # data: data/mfcc_23_pitch/voxceleb1o2_train/ 
  # 如果只传入1个参数，也就是$aug_data_dir==""，则用$subset_data覆盖$data目录
	[ ! -d $aug_data_dir ] && ${SUBTOOLS}/kaldi/utils/combine_data.sh $aug_data_dir $data $subset_data
fi

echo "All done."
