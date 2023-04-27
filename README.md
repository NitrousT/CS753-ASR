# AV-HuBERT (Audio-Visual Hidden Unit BERT) for ReVISE

## Introduction
AV-HuBERT is a self-supervised representation learning framework for audio-visual speech. It achieves state-of-the-art results in lip reading, ASR and audio-visual speech recognition on the LRS3 audio-visual speech benchmark.

<img src="../assets/model.png" alt="Image" width="400">

## Pre-trained and fine-tuned models

Please find the checkpoints [here](http://facebookresearch.github.io/av_hubert)

## Installation
First, create a conda virtual environment and activate it:
```
conda create -n avhubert python=3.8 -y
conda activate avhubert
```
Then, clone this directory:
```
git clone https://github.com/facebookresearch/av_hubert.git
cd avhubert
git submodule init
git submodule update
```

Lastly, install Fairseq and the other packages:
```
pip install -r requirements.txt
cd fairseq
pip install --editable ./
```
## Load a pretrained model
```sh
$ cd avhubert
$ python
>>> import fairseq
>>> import hubert_pretraining, hubert
>>> ckpt_path = "/path/to/the/checkpoint.pt"
>>> models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
>>> model = models[0]
```

# Data preparation

### Installation
To preprocess, you need some additional packages:
```
pip install -r requirements.txt
```

## Datasets and Preprocessing
Download and decompress the [data](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html). Assume the data directory is `${lrs3}`, which contains three folders (`pretrain,trainval,test`).

Download and decompress the [data](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html). Assume the data directory is `${vox}`, which contains two folders (`dev,test`).

For detailed instructions on how to prepare data, follow the README.md instructions inside the preparation folder.


# AV-HuBERT Label Preparation

This folder contains scripts for preparing AV-HUBERT labels from tsv files, the
steps are:
1. feature extraction
2. k-means clustering
3. k-means application

## Installation
To prepare labels, you need some additional packages:
```
pip install -r requirements.txt
```

## Data preparation

`*.tsv` files contains a list of audio, where each line is the root, and
following lines are the subpath and number of frames of each video and audio separated by `tab`:
```
<root-dir>
<id-1> <video-path-1> <audio-path-1> <video-number-frames-1> <audio-number-frames-1>
<id-2> <video-path-2> <audio-path-2> <video-number-frames-2> <audio-number-frames-2>
...
```
See [here](../preparation/) for data preparation for LRS3 and VoxCeleb2. 

## Feature extraction

### MFCC feature
Suppose the tsv file is at `${tsv_dir}/${split}.tsv`. To extract 39-D
mfcc+delta+ddelta features for the 1st iteration AV-HuBERT training, run:
```sh
python dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir}
```
This would shard the tsv file into `${nshard}` and extract features for the
`${rank}`-th shard, where rank is an integer in `[0, nshard-1]`. Features would
be saved at `${feat_dir}/${split}_${rank}_${nshard}.{npy,len}`.


### AV-HuBERT feature
To extract features from the `${layer}`-th transformer layer of a trained
AV-HuBERT model saved at `${ckpt_path}`, run:
```sh
python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir} --user_dir `pwd`/../
```
Features would also be saved at `${feat_dir}/${split}_${rank}_${nshard}.{npy,len}`.

- if out-of-memory, decrease the chunk size with `--max_chunk`


## K-means clustering
To fit a k-means model with `${n_clusters}` clusters on 10% of the `${split}` data, run
```sh
python learn_kmeans.py ${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 0.1
```
This saves the k-means model to `${km_path}`.

- set `--precent -1` to use all data
- more kmeans options can be found with `-h` flag


## K-means application
To apply a trained k-means model `${km_path}` to obtain labels for `${split}`, run
```sh
python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
```
This would extract labels for the `${rank}`-th shard out of `${nshard}` shards
and dump them to `${lab_dir}/${split}_${rank}_${shard}.km`


Finally, merge shards for `${split}` by running
```sh
for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/${split}_${rank}_${nshard}.km
done > $lab_dir/${split}.km
```
and create a dictionary of cluster indexes by running
```sh
for i in $(seq 1 $((n_cluster-1)));do 
    echo $i 10000
done > $lab_dir/dict.{mfcc,km}.txt
```


## Clustering on slurm
If you are on slurm, you can combine the above steps (feature extraction + K-means clustering + K-means application) by:

- MFCC feature cluster:
```sh
python submit_cluster.py --tsv ${tsv_dir} --output ${lab_dir} --ncluster ${n_cluster} \
  --nshard ${nshard} --mfcc --percent 0.1
```

- AV-HuBERT feature cluster:
```sh
python submit_cluster.py --tsv ${tsv_dir} --output ${lab_dir} --ckpt ${ckpt_path} --nlayer ${layer} \
  --ncluster ${n_cluster} --nshard ${nshard} --percent 0.1
```

This would  dump labels to `${lab_dir}/{train,valid}.km`.

# Inference (Demo)

The inference pipeline is made simple an quick:

```sh
python adams_apple.py
```
Note: Make sure that ckpt_model, mouth_roi_path and user_dir variables are modified according to your system. We have simply a placeholder path locations.

Download the sample test video:
```sh
!mkdir -p /content/data/misc/
!wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O /content/data/misc/shape_predictor_68_face_landmarks.dat.bz2
!bzip2 -d /content/data/misc/shape_predictor_68_face_landmarks.dat.bz2
!wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O /content/data/misc/20words_mean_face.npy

```
Download the video from the internet
```sh
!wget --content-disposition https://dl.fbaipublicfiles.com/avhubert/demo/avhubert_demo_video_8s.mp4 -O /content/data/clip.mp4
```

# Finetunning

Suppose `{train,valid}.tsv` are saved at `/path/to/data`, `{train,valid}.wrd`
are saved at `/path/to/labels`, the configuration file is saved at `/path/to/conf/conf-name`.

To fine-tune a pre-trained HuBERT model at `/path/to/checkpoint`, run:
```sh
$ cd avhubert
$ fairseq-hydra-train --config-dir /path/to/conf/ --config-name conf-name \
  task.data=/path/to/data task.label_dir=/path/to/label \
  task.tokenizer_bpe_model=/path/to/tokenizer model.w2v_path=/path/to/checkpoint \
  hydra.run.dir=/path/to/experiment/finetune/ common.user_dir=`pwd`