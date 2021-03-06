# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-01-05)

import sys
import os
import logging
import argparse
import traceback

# subtools = '/data/lijianchen/workspace/sre/subtools'
subtools = os.getenv('SUBTOOLS')
sys.path.insert(0, '{}/pytorch'.format(subtools))

from libs.egs.samples import ChunkSamples
from libs.egs.kaldi_dataset import KaldiDataset
import libs.support.kaldi_common as kaldi_common


"""
Get chunk egs for sre and lid ... which use the xvector framework.
"""

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [ %(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ]\n#### %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    # Start
    parser = argparse.ArgumentParser(
        description="""Split data to chunk-egs.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    # Options
    # valid: validation set
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="A fixed chunk size.")

    parser.add_argument("--sample-type", type=str, default='speaker_balance',
                        choices=["speaker_balance", "sequential"],
                        help="The sample type for trainset.")

    parser.add_argument("--chunk-num", type=int, default=-1,
                        help="Define the avg chunk num. -1->suggestion"
                        "（total_num_chunks / num_spks * scale）, 0->max_num_chunks, int->int")

    parser.add_argument("--scale", type=float, default=1.5,
                        help="The scale for --chunk-num:-1.")

    parser.add_argument("--overlap", type=float, default=0.1,
                        help="The scale of overlap to generate chunks.")

    parser.add_argument("--drop-last", type=str, action=kaldi_common.StrToBoolAction,
                        default=False, choices=["true", "false"],
                        help="Drop the last sample of every utterance or not.")

    parser.add_argument("--valid-dir", type=str, default="", help="A kaldi datadir.")

    parser.add_argument("--valid-split-from-trainset", type=str, action=kaldi_common.StrToBoolAction,
                        default=True, choices=["true", "false"], help="")

    parser.add_argument("--valid-split-type", type=str, default='--total-spk',
                        choices=["--default", "--per-spk", "--total-spk"],
                        help="Get the valid samples or not.")

    parser.add_argument("--valid-num-utts", type=int, default=1024,
                        help="The num utts to split for valid. 1024 for --total-spk.")

    parser.add_argument("--valid-sample-type", type=str, default='every_utt',
                        choices=["speaker_balance", "sequential", "every_utt", "full_length"],
                        help="The sample type for valid set.")

    parser.add_argument("--valid-chunk-num", type=int, default=2,
                        help="define the avg chunk num. -1->suggestion"
                             "（max / num_spks * scale）, 0->max, int->int"
                             "chunk num of every validation set utterance ")

    parser.add_argument("--valid-scale", type=float, default=1.5,
                        help="The scale for --valid-chunk-num:-1.")

    parser.add_argument("--seed", type=int, default=1024,
                        help="random seed")

    parser.add_argument("--expected-files", type=str, default="utt2spk,spk2utt,feats.scp,utt2num_frames")

    # Main
    parser.add_argument("data_dir", metavar="data-dir",
                        type=str, help="A kaldi datadir.")
    parser.add_argument("save_dir", metavar="save-dir", type=str,
                        help="The save dir of mapping file of chunk-egs.")

    # End
    print(' '.join(sys.argv))
    args = parser.parse_args()

    return args


def get_chunk_egs(args):
    logger.info("Load kaldi datadir {0}".format(args.data_dir))
    expected_files = args.expected_files.split(',')
    dataset = KaldiDataset.load_data_dir(args.data_dir, expected_files=expected_files)
    if "utt2spk_int" not in expected_files:
        dataset.generate("utt2spk_int")

    if args.valid_dir != "":
        valid = KaldiDataset.load_data_dir(args.valid_dir, expected_files=expected_files)
        if "utt2spk_int" not in expected_files:
            valid.generate("utt2spk_int", dataset.spk2int)
        trainset = dataset
    elif args.valid_split_from_trainset is True:
        logger.info("Split valid dataset from {0}".format(args.data_dir))
        if args.valid_num_utts > len(dataset) // 10:
            logger.info("Warning: the --valid-num-utts ({0}) of valid set is out "
                        "of 1/10 * num of original dataset ({1}). Suggest to be less.".format(
                            args.valid_num_utts, len(dataset)))
        trainset, valid = dataset.split(args.valid_num_utts, args.valid_split_type, seed=args.seed)
    else:
        trainset = dataset


    # f = open("/home/lijianchen/workspace/sre/spk2utt_{}.txt".format(args.data_dir[-11:-7]), 'w')
    # for key in trainset.spk2utt.keys():
    #     f.write(key + ' ' + " ".join(trainset.spk2utt[key]) + '\n')
    # f.close
    # f = open("/home/lijianchen/workspace/sre/utt2spk_{}.txt".format(args.data_dir[-11:-7]), 'w')
    # for key in trainset.utt2spk.keys():
    #     f.write(key + ' ' + trainset.utt2spk[key] + '\n')
    # f.close
    # f = open("/home/lijianchen/workspace/sre/spk2utt_val_{}.txt".format(args.data_dir[-11:-7]), 'w')
    # for key in valid.spk2utt.keys():
    #     f.write(key + ' ' + " ".join(valid.spk2utt[key]) + '\n')
    # f.close
    # f = open("/home/lijianchen/workspace/sre/utt2spk_val_{}.txt".format(args.data_dir[-11:-7]), 'w')
    # for key in valid.utt2spk.keys():
    #     f.write(key + ' ' + valid.utt2spk[key] + '\n')
    # f.close



    logger.info("Generate chunk egs with chunk-size={0}.".format(args.chunk_size))
    # 按照一定的overlap将每个utterance分割成chunk
    trainset_samples = ChunkSamples(trainset, args.chunk_size, chunk_type=args.sample_type,
                                    chunk_num_selection=args.chunk_num, scale=args.scale,
                                    overlap=args.overlap, drop_last=args.drop_last, seed=args.seed)

    if args.valid_dir != "" or args.valid_split_from_trainset is True:
        valid_sample = ChunkSamples(valid, args.chunk_size, chunk_type=args.valid_sample_type,
                                    chunk_num_selection=args.valid_chunk_num,
                                    scale=args.valid_scale, overlap=args.overlap,
                                    drop_last=args.drop_last, seed=args.seed)

    logger.info("Save mapping file of chunk egs to {0}".format(args.save_dir))
    if not os.path.exists("{0}/info".format(args.save_dir)):
        os.makedirs("{0}/info".format(args.save_dir))

    trainset_samples.save("{0}/train.egs.csv".format(args.save_dir))

    if args.valid_dir != "" or args.valid_split_from_trainset is True:
        valid_sample.save("{0}/validation.egs.csv".format(args.save_dir))

    with open("{0}/info/num_frames".format(args.save_dir), 'w') as writer:
        writer.write(str(trainset.num_frames))

    with open("{0}/info/feat_dim".format(args.save_dir), 'w') as writer:
        writer.write(str(trainset.feat_dim))

    with open("{0}/info/num_targets".format(args.save_dir), 'w') as writer:
        writer.write(str(trainset.num_spks))

    logger.info("Generate egs from {0} done.".format(args.data_dir))


def main():
    args = get_args()

    try:
        get_chunk_egs(args)
    except BaseException as e:
        # Look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
