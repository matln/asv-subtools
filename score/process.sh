#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-11-07)

# add a function and add a mark in get_params() and add a config for which needs 'get_resource' in the top script and that's all

function check(){
	varset=$1
	support=$2
	name=$3
	
	for x in ${varset//-/ }; do
		lable=0
		for i in $support; do
			[ "$x" == "$i" ] && lable=1 && break
		done
		[ "$lable" == 0 ] && echo "[exit] Don't support [ $x ] in [ $varset ] $name and support [ $support ] only." && exit 1
	done
	
	return 0
}

function readconf(){
	[ $# != 2 ] && echo "[exit] readconf(): num of params != 2" 1>&2 && exit 1
	local key=$1
	local conf=$2
	
	[ ! -f $conf ] && echo "[exit] $conf is not exist, please check provided dataset" 1>&2 && exit 1
	value=$(awk -v key=$key '{if($1==key) print $2}' $conf)
	[ "$value" == "" ] && echo "[exit] the value of key [$key] in $conf is not exist, please check data config about $key" 1>&2 && exit 1
	echo $value
	return 0
}

function writeconf(){
	[ $# != 3 ] && return 0
	local key=$1
	local value=$2
	local conf=$3
	
	[ -f $conf ] && \
    sed -i '/^'"$key"' /d' $conf
	
	[ -f $conf ] && \
    echo "$key $value" >> $conf
	return 0
}

# 确认 outfile 是否已经在 existed_outfile list 中存在
function findoutfile(){
	[ $# != 2 ] && echo "[exit] findlist(): num of params != 2" 1>&2 && exit 1
  local outfile=$1
	existed_outfile=$(cat "$2")
	
	num=$(echo -e "$existed_outfile\n$existed_outfile\n$outfile" | sort | uniq -u | wc -l )
	echo "$num"
	return 0
}

# 某些 process 需要先进行其他的一些 process，例如，submean_process="lda-getmean"
# 我们需要先 lda 降维，再计算 global_mean，最后才是 subtract_mean 操作
# 通过 get_resource 函数得到当前 process 所需要的前序 process 的结果文件 
function get_resource(){
	local prekey=$1
	local conf=$2
	
	tmp_data_conf=$(readconf "${prekey}_data_conf" $conf)
	tmp_string=$(readconf "${prekey}_process" $tmp_data_conf)
	tmp_dir=$(readconf "vectordir" $tmp_data_conf)
	out=$(process $tmp_data_conf $tmp_string)
	
	echo $tmp_dir/$out
	
	return 0
}

# 获取 process 执行的参数
function get_params(){
	local process=$1
	local conf=$2
	local vectorfile=$3

	local data
	data=$(readconf "data" $conf)
	local dir
	dir=$(readconf "vectordir" $conf)
  local outfile

	case $process in
		mean)
			outfile="spk_${vectorfile%.*}_${process}.ark"
			string="${data}/spk2utt ${dir}/$vectorfile ${dir}/$outfile ${dir}/num_utts.ark";;
		getmean)
			outfile="${vectorfile%.*}.global.vec"
			string="${dir}/$vectorfile ${dir}/$outfile";;
		submean)
			global_mean=$(get_resource submean $conf)
			outfile="${vectorfile%.*}_${process}.ark"
			string="$global_mean ${dir}/$vectorfile ${dir}/$outfile";;
		norm)
			outfile="${vectorfile%.*}_${process}.ark"
			string="${dir}/$vectorfile ${dir}/$outfile";;
		lda)
			lda_mat=$(get_resource lda $conf)
			outfile="${vectorfile%.*}_${process}${clda}.ark"
			string="$lda_mat ${dir}/$vectorfile ${dir}/$outfile";;
		trainlda)
			outfile="transform_$clda.mat"
			string="${dir}/$vectorfile ${data}/utt2spk ${dir}/$outfile";;
		whiten)
			whiten_mat=$(get_resource whiten $conf)
			outfile="${vectorfile%.*}_${process}.ark"
			string="$whiten_mat ${dir}/$vectorfile ${dir}/$outfile";;
		trainwhiten)
			outfile="zca_whiten.mat"
			string="${dir}/$vectorfile ${dir}/$outfile";;
		trainpcawhiten)
			outfile="pca_whiten.mat"
			string="${dir}/$vectorfile ${dir}/$outfile";;
		trainplda)
			outfile="plda"
			string="$data/spk2utt $dir/$vectorfile $dir/$outfile";;
		trainaplda)
			plda=$(get_resource plda $conf)
			outfile="aplda"
			string="$plda $dir/$vectorfile $dir/$outfile";;
		*)echo "[exit] Do not support $process process now." 1>&2 && exit 1;;
	esac
	
	echo $outfile $string
	return 0
}

function process(){
	local conf=$1
	local process_string=$2
	
	current_file=$(readconf "vectorfile" "$conf")
  local dir
	dir=$(readconf "vectordir" "$conf")
	
	for the_process in ${process_string//-/ }; do
		doit=1
    # 通过 source 或 . 加载此脚本，所以可以使用 $lda 等变量替换
		[[ "$lda" != "true" && "$the_process" == "lda" ]] && doit=0
		[[ "$submean" != "true" && "$the_process" == "submean" ]] && doit=0
		[[ "$whiten" != "true" && "$the_process" == "whiten" ]] && doit=0
		
		if [ "$doit" == 1 ]; then
			tmp=$(get_params $the_process $conf $current_file)
			current_file=$(echo "$tmp" | awk '{print $1}')
			params=$(echo "$tmp" | awk '{$1="";print $0}')
			exist=$(findoutfile "$dir/$current_file" "$processed")  # 1 -> do not exist  3 - > has been processed

			[[ ! -f "$dir/$current_file" || "$process_force_clear" == "true" ]] && [[ "$exist" == 1 ]] && \
        $the_process $params
			echo "$dir/$current_file" >> "$processed"
		fi
	done
	
	echo "$current_file"
	return 0
}


############################################################################
function mean(){
	local spk2utt=$1
	local vectorfile=$2
	local outfile=$3
	local num_utts=$4
	
	specifier=ark
	[ "${#vectorfile#*.}" == "scp" ] && specifier=scp
	
	ivector-mean ark:"$spk2utt" $specifier:"$vectorfile" ark:"$outfile" ark,t:"$num_utts" || exit 1
	return 0
}

function getmean(){
# compute global mean.vector for substract mean.vector
	local vectorfile=$1
	local outmean=$2
	
	specifier=ark
	[ "${vectorfile##*.}" == "scp" ] && specifier=scp
	
	ivector-mean $specifier:"$vectorfile" "$outmean" || exit 1
	return 0
}

function submean(){
# substract global mean.vector
	local mean=$1
	local vectorfile=$2
	local outfile=$3
	
	specifier=ark
	[ "${vectorfile##*.}" == "scp" ] && specifier=scp
	
	ivector-subtract-global-mean "$mean" $specifier:"$vectorfile" ark:"$outfile" || exit 1
	return 0
}

function norm(){
	local vectorfile=$1
	local outfile=$2
	
	specifier=ark
	[ "${vectorfile##*.}" == "scp" ] && specifier=scp
	
	ivector-normalize-length --scaleup=false $specifier:"$vectorfile" ark:"$outfile" || exit 1
	return 0
}

function transform(){
# an implement of the interface for any matrix to transform data
	local mat=$1
	local vectorfile=$2
	local outfile=$3
	
	specifier=ark
	[ "${vectorfile##*.}" == "scp" ] && specifier=scp
	
	ivector-transform "$mat" $specifier:"$vectorfile" ark:"$outfile" || exit 1
	return 0
}

function trainlda(){
	local vectorfile=$1
	local utt2spk=$2
	local outfile=$3
	
	specifier=ark
	[ "${vectorfile##*.}" == "scp" ] && specifier=scp
	
	ivector-compute-lda --dim="$clda" --total-covariance-factor=0.1 $specifier:"$vectorfile" ark:"$utt2spk" "$outfile" || exit 1
	return 0
}

function lda(){
	transform "$@"
	return 0
}

function trainwhiten(){
# ZCA whitening
	local trainfile=$1
	local outmat=$2
	
	train_specifier=ark
	[ "${trainfile##*.}" == "scp" ] && train_specifier=scp
	
	[ ! -f $trainfile.txt ] && copy-vector $train_specifier:$trainfile ark,t:$trainfile.txt
	
	# should print information to terminal with 1>&2 when using python or will be error
	python3 ${SUBTOOLS}/score/whiten/train_ZCA_Whitening.py --ark-format=true $trainfile.txt $outmat 1>&2  || exit 1
	return 0
}

function trainpcawhiten(){
# PCA whitening 
	local trainfile=$1
	local outmat=$2
	
	train_specifier=ark
	[ "${trainfile##*.}" == "scp" ] && train_specifier=scp
	
	est-pca --read-vectors=true $train_specifier:$trainfile $outmat || exit 1
	return 0
}

function whiten(){
	transform "$@"
	return 0
}

function trainplda(){
	local spk2utt=$1
	local vectorfile=$2
	local outfile=$3
	
	specifier=ark
	[ "${vectorfile##*.}" == "scp" ] && specifier=scp
	
	ivector-compute-plda ark:"$spk2utt" $specifier:"$vectorfile" "$outfile" || exit 1
	
	return 0
}

function trainaplda(){
	local plda=$1
	local vectorfile=$2
	local outfile=$3
	
	specifier=ark
	[ "${vectorfile##*.}" == "scp" ] && specifier=scp
	
	ivector-adapt-plda --within-covar-scale=$within_covar_scale --between-covar-scale=$between_covar_scale \
		--mean-diff-scale=$mean_diff_scale $plda $specifier:"$vectorfile" "$outfile" || exit 1
		
	return 0
}
