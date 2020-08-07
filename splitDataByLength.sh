#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2019-04-25)

outputdir= # If NULL, default $datadir/split*
force=false # If true, split again whatever

. ${SUBTOOLS}/parse_options.sh
# . subtools/path.sh

set -e

if [[ $# != 2 ]]; then
  echo "[exit] Num of parameters is not equal to 2"
  echo "usage:$0 [--outputdir ""] [--force false|true] <data-dir> <num-nj>"
  exit 1
fi

data=$1
nj=$2

[ ! -d $data ] && echo "[exit] No such dir $data." && exit 1

for x in feats.scp vad.scp; do
  [ ! -f $data/$x ] && echo "[exit] expect $data/$x to exist." && exit 1
done

[ "$outputdir" == "" ] && outputdir=$data/split${nj}order
[ "$force" != "true" ] && [ -f "$outputdir/.done" ] && echo "[Note] Do not split $data again..." && exit 0

echo "Split $data with $nj nj according to length-order ..."

${SUBTOOLS}/get_utt2num_frames_from_vad.sh --nosil true --nj $nj $data

mkdir -p $outputdir

sort -r -n -k 2 $data/utt2num_frames.nosil > $outputdir/utt2num_frames.nosil.order

tot_num=$(wc -l $outputdir/utt2num_frames.nosil.order | awk '{print $1}')

[[ "$tot_num" -lt "$nj" ]] && echo "nj $nj is too large for $tot_num utterances." && exit 1

num_frames=$(awk '{a=a+$2}END{print a}' $outputdir/utt2num_frames.nosil.order ) 

average_frames=$num_frames
[ "$nj" != 1 ] && average_frames=$[$num_frames/$nj + 1]

echo -e "num_frames:$num_frames\naverage_frames:$average_frames"

for i in $(seq $nj); do
  mkdir -p $outputdir/$i
  > $outputdir/$i/utt2num_frames.nosil.order
done

# split 
# 假如nj=16，想象有16个水桶，line代表第几次倒水，倒水的量为b[line]，
# 要求，先从第一个桶开始，第一个桶倒b[1]+b[2]的水，然后倒第二个桶，只要当前水桶的量比前面的水桶多一点，
# 就开始倒下一个水桶，也就是后面的桶倒完水后都要比前面的桶多一点。
# 完成后再反过来，从第16只水桶开始往前倒，此时要求倒完水后比后面的桶多一点
# NR-line+avoid_endless_loop>=i:
#   NR-line 表示剩余倒水的次数，一开始avoid_endless_loop=1，当该式子小于i时，表明
#   剩余倒水的次数已经不够了，甚至不足以给剩下的水桶各倒一次。所以跳出循环，先从先前的水桶开始倒，
#   因为此时越往前，水桶的水量越小，所先尽量从量小的水桶开始倒。
#   从前面开始倒的话，如果倒完后都超过了mean，此时是不会倒水的。
#   只能增大avoid_endless
awk -v nj=$nj -v mean=$average_frames -v dir=$outputdir '{a[NR]=$1;b[NR]=$2;}
  END{
    line=1;
    max=b[line]+1;
    avoid_endless_loop=1;
    while(line<=NR){
      out=0;
      for(i=1;i<=nj;i++){
        while(c[i]<max && c[i]+b[line]<mean){
          c[i]=c[i]+b[line];
          print a[line],b[line] >> dir"/"i"/utt2num_frames.nosil.order"
          out=out+1;
          line=line+1;
          if(line>NR){break;}
        }
        if(line>NR){break;}
        if(c[i]>=max){max=c[i];}
      }
      for(i=nj;i>=1;i--){
        while(c[i]<=max && c[i]<mean && NR-line+avoid_endless_loop>=i){
          c[i]=c[i]+b[line];
          print a[line],b[line] >> dir"/"i"/utt2num_frames.nosil.order"
          out=out+1;
          line=line+1;
          if(line>NR){break;}
        }
        if(line>NR){break;}
        if(c[i]>=max){max=c[i];}    
      }
      if(out==0){
        avoid_endless_loop=avoid_endless_loop+1;
      }
      else if(avoid_endless_loop>1){
        avoid_endless_loop=avoid_endless_loop-1;
      }
    }
  }' $outputdir/utt2num_frames.nosil.order

# filter
for i in $(seq $nj); do
  ${SUBTOOLS}/filterDataDir.sh --check false $data $outputdir/$i/utt2num_frames.nosil.order $outputdir/$i/ >/dev/null
done

> $outputdir/.done # a mark file

rm -rf $outputdir/*/.backup

echo "Split done."




