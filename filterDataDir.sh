#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2019-02-17)

# This script creates a subset of data according to the utterance list file, see also kaldi/utils/subset_data_dir.sh
# If split_aug=true, then the utterance list file will be expanded to include utterance with suffix in $aug_suffixes

f=1 # field of utt-id in id-file
exclude=false
split_aug=false
aug_suffixes="reverb noise music babble"
check=true

. ${SUBTOOLS}/parse_options.sh

if [[ $# != 3 ]]; then
  echo "[exit] Num of parameters is not equal to 3"
  echo "$0 [--check true|false] [--exclude false|true] [--f 1] <in-data-dir> <id-list> <out-data-dir>"
  exit 1
fi

indata=$1
idlist=$2
outdata=$3

[ ! -d "$indata" ] && echo "[exit] No such dir $indata" && exit 1
[ ! -f "$idlist" ] && echo "[exit] No such file $idlist" && exit 1
[ "$check" == "true" ] && [ -d "$outdata" ] && echo "[exit] $outdata is exist." && exit 1
mkdir -p "$outdata"

if [ "$split_aug" == "true" ]; then
  [ "$f" -ne "1" ] && echo "Expected -f=1 with utt-id to use split_aug" && exit 1

  cat "$idlist" > "${idlist}"_aug
  for aug_suffix in $aug_suffixes; do
    awk -v suffix="$aug_suffix" '{print $1"-"suffix, $2}' $idlist >> ${idlist}_aug
  done
  idlist=${idlist}_aug
fi

exclude_string=""
[[ "$exclude" == "true" ]] && exclude_string="--exclude"

for x in wav.scp utt2spk feats.scp utt2num_frames vad.scp utt2dur text; do
  [ -f "$indata/$x" ] && awk -v f=$f '{print $f}' $idlist | ${SUBTOOLS}/kaldi/utils/filter_scp.pl $exclude_string - $indata/$x > $outdata/$x
done

# include: utt2spk_to_spk2utt.pl
${SUBTOOLS}/kaldi/utils/fix_data_dir.sh $outdata

rm -rf ${idlist}_aug $outdata/.backup
echo "Filter $outdata done."
