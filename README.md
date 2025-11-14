<div align="center">
<h1>Improve Timbre Consistency in Cross Domain Singing Voice Conversion</h1>

</div>

Our Project is inspired by [CoMoSVC](https://github.com/Grace9994/CoMoSVC).


## Environment
You can set up your Conda environment using the following command:

```shell
conda env create -f environment.yaml
```

## Download the Checkpoints
### 1. m4singer_hifigan

You should first download [m4singer_hifigan](https://drive.google.com/file/d/10LD3sq_zmAibl379yTW5M-LXy2l_xk6h/view) and then unzip the zip file by
```shell
unzip m4singer_hifigan.zip
```
The checkpoints of the vocoder will be in the `m4singer_hifigan` directory

### 2. ContentVec
You should download the checkpoint [ContentVec](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) and the put it in the `Content` directory to extract the content feature.

### 3. m4singer_pe
You should download the pitch_extractor checkpoint of the [m4singer_pe](https://drive.google.com/file/d/19QtXNeqUjY3AjvVycEt3G83lXn2HwbaJ/view) and then unzip the zip file by 

```shell
unzip m4singer_pe.zip
```

### 4. samresnet34
You should download the pitch_extractor checkpoint of the [samresnet34](https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet34_ft.zip) and then unzip the zip file by 

```shell
voxblink2_samresnet34_ft.zip
```
## Dataset Preparation 

You should first create the folders by

```shell
mkdir dataset_raw
mkdir dataset
```
You can refer to different preparation methods based on your needs.

Preparation With Slicing can help you remove the silent parts and slice the audio for stable training.


### 0. Preparation With Slicing

Please place your original dataset in the `dataset_slice` directory.

The original audios can be in any waveformat which should be specified in the command line. You can designate the length of slices you want, the unit of slice_size is milliseconds. The default wavformat and slice_size is mp3 and 10000 respectively.

```shell
python preparation_slice.py -w your_wavformat -s slice_size
```

### 1. Preparation Without Slicing

You can just place the dataset in the `dataset_raw` directory with the following file structure:

```
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```


##  Preprocessing

### 1. Resample to 24000Hz and mono

```shell
python preprocessing1_resample.py -n num_process
```
num_process is the number of processes, the default num_process is 5.

### 2. Split the Training and Validation Datasets, and Generate Configuration Files.

```shell
python preprocessing2_flist.py
```


### 3. Generate Features

```shell
python preprocessing3_feature.py -c your_config_file -n num_processes 
```
### 4. Extract Timbre Features

```shell
python extract_spk_embd.py --base_dir ./dataset_raw --result_dir ./dataset
```
### 5. Extract F0 Features

```shell
python extract_spk_embd.py --dataset_dir ./dataset
```

## Training

### 1. Set up config file

### 2. Train the  Model

```shell
bash train.sh
```
The checkpoints will be saved in the `logs` directory

## Inference
You should put the audios you want to convert under the `raw` directory firstly.

### Inference by the Teacher Model

```shell
python simple_inference.py \
    --source_audio path_to_your_source_music \
    --target_speaker path_to_your_target_timbre \
    -k 0 \
    -m path_to_your_model_ckpt \
    -c path_to_your_model_config
```

