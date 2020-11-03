#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2019-06-03)
#                     (Author: Hao Lu  2020-03-13 "Add diarization")

nj=16
cmd="run.pl"
stage=1
cmn=true
vad=false
cmn_window=300
model=final.params
split_type=order
use_gpu=false
gpu_id=""
force=false
sleep_time=3
nnet_config=config/nnet.config

# Diarization
sliding=false
window=1.5
period=0.75
min_segment=0.5
hard_min=false

echo "$0 $@"

set -e 

# if [ -f subtools/path.sh ]; then . subtools/path.sh; fi
. "${SUBTOOLS}"/parse_options.sh || exit 1;

if [[ $# != 3 ]]; then
  echo "[exit] Num of parameters is not equal to 3"
  echo "usage:$0 <model-dir> <data-dir> <output-dir>"
  exit 1
fi

srcdir=$1
data=$2
dir=$3

# Check
mkdir -p $dir/log

num=0

[ -s "$dir"/xvector.scp ] && num=$(grep -E "ERROR|Error" $dir/log/extract.*.log | wc -l)

[[ "$force" != "true" && -s $dir/xvector.scp && $num == 0 ]] && echo "Do not extract xvectors of [ $data ] to [ $dir ] again with force=$force." && exit 0

rm -rf $dir/log/* # It is important for the checking.

# Start

# Diarization
if [ "$sliding" == "true" ]; then
  sub_data=$dir/subsegments_data
  mkdir -p $sub_data
  # Set up sliding-window subsegments
    if $hard_min; then
      awk -v min=$min_segment '{if($4-$3 >= min){print $0}}' $data/segments \
          > $dir/pruned_segments
      segments=$dir/pruned_segments
    else
      segments=$data/segments
    fi

    [ ! -f $segments ] && echo "Expected $segments to exist." && exit 1

    subtools/kaldi/utils/data/get_uniform_subsegments.py \
        --max-segment-duration=$window \
        --overlap-duration=$(perl -e "print ($window-$period);") \
        --max-remaining-duration=$min_segment \
        --constant-duration=True \
        $segments > $dir/subsegments
    subtools/kaldi/utils/data/subsegment_data_dir.sh $data \
        $dir/subsegments $sub_data

  # Creat visual vad
  subtools/createVisualVad.sh $sub_data
  data=$sub_data
fi

for f in $srcdir/$model $srcdir/$nnet_config $data/feats.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

case $split_type in
    default)
      "${SUBTOOLS}"/kaldi/utils/split_data.sh --per-utt $data $nj
      sdata=$data/split${nj}utt/JOB
      ;;
    order)
      "${SUBTOOLS}"/splitDataByLength.sh --vad $vad $data $nj
      sdata=$data/split${nj}order/JOB
      ;;
    *)
      echo "[exit] Do not support $split_type split-type" && exit 1
      ;;
esac

echo "$0: extracting xvectors for $data"


# Set up the features
if [ "$cmn" == "true" ]; then
  if [ "$vad" == "true" ]; then
    feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=$cmn_window \
           scp:${sdata}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:${sdata}/vad.scp ark:- |"
  else
    feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=$cmn_window scp:${sdata}/feats.scp ark:- |"
  fi
else
  if [ "$vad" == "true" ];then
    feats="ark:select-voiced-frames scp:${sdata}/feats.scp scp,s,cs:${sdata}/vad.scp ark:- |"
  else
    feats="ark:copy-feats scp:${sdata}/feats.scp ark:- |"
  fi
fi
  output="ark:| copy-vector ark:- ark,scp:$dir/xvector.JOB.ark,$dir/xvector.JOB.scp"

if [ $stage -le 1 ]; then
  echo "$0: extracting xvectors from pytorch nnet"
  trap "${SUBTOOLS}/linux/kill_pid_tree.sh --show true $$ && echo -e '\nAll killed\n' && exit 1" INT
  if $use_gpu; then
    pids=""
    if [ "$gpu_id" == "" ]; then
      # auto choice gpu device

      # run.pl --gpu 1 $max_job_run 会设置为gpu的数量，但这里每个run.pl只有一个子进程，单独的run.pl并没有并行
      #   run.pl 这里只起到了输出log文件的作用
      for g in $(seq $nj); do
        $cmd --gpu 1 ${dir}/log/extract.$g.log \
          python3 ${SUBTOOLS}/pytorch/pipeline/onestep/extract_embeddings.py --use-gpu=$use_gpu --gpu-id="$gpu_id" \
                  --nnet-config=$srcdir/$nnet_config \
                  "$srcdir/$model" "`echo $feats | sed s/JOB/$g/g`" "`echo $output | sed s/JOB/$g/g`" || exit 1 &
        sleep $sleep_time
        pids="$pids $!"
      done
    else
      # 自己指定使用的gpu，循环使用指定的gpu
      gpu_id=$(echo "$gpu_id" | sed 's/,/ /g')  # default delimiter is comma, like: --gpu-id=3,4,5
      arr_gpu=($gpu_id)
      num_gpu=${#arr_gpu[@]}
      for g in $(seq $nj); do
        index=$[($g - 1) % $num_gpu]
        gpu_id=${arr_gpu[$index]}
        $cmd --gpu 1 ${dir}/log/extract.$g.log \
          python3 ${SUBTOOLS}/pytorch/pipeline/onestep/extract_embeddings.py --use-gpu=$use_gpu --gpu-id="$gpu_id" \
                  --nnet-config=$srcdir/$nnet_config \
                  "$srcdir/$model" "`echo $feats | sed s/JOB/$g/g`" "`echo $output | sed s/JOB/$g/g`" || exit 1 &
        sleep $sleep_time
        pids="$pids $!"
      done
    fi
    trap "${SUBTOOLS}/linux/kill_pid_tree.sh --show true $pids && echo -e '\nAll killed' && exit 1" INT
    wait
  else
    $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
        python3 ${SUBTOOLS}/pytorch/pipeline/onestep/extract_embeddings.py --use-gpu="false" \
                --nnet-config=$srcdir/$nnet_config \
                "$srcdir/$model" "$feats" "$output" || exit 1;
  fi

  num=$(grep -E "ERROR|Error" $dir/log/extract.*.log | wc -l)
  [ $num -gt 0 ] && echo "There are some ERRORS in $dir/log/extract.*.log." && exit 1
fi

if [ $stage -le 2 ]; then
  echo "$0: combining xvectors across jobs"
  for j in $(seq $nj); do cat $dir/xvector.$j.scp; done >$dir/xvector.scp || exit 1;
fi

echo "Embeddings of [ $data ] has been extracted to [ $dir ] done."

exit 0
