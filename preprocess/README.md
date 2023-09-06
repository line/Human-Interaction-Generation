# Preparing dataset for interaction generation

## Preparing Dataset

Download [NTU RGB+D and NTU RGB+D 120 dataset](https://rose1.ntu.edu.sg/dataset/actionRecognition/) and put them in the following directory.

```
dataset_dir
└───video_dir
│   └───nturgb+d_rgb_s001
│   └───nturgb+d_rgb_s002
│   └───...
└───output
└───preprocessed
```

## Extract and Preprocess 3D pose sequences from videos

1. Go to `./Extract3Dpose/simple_romp` and Run following commands.

```
1. Get interaction videos.
python get_interaction_videos.py --dataset_path /path/to/dataset_dir
2. Run bev for all interaction videos.
python run_bev_for_interaction_videos.py --id 1 --dataset_path /path/to/dataset_dir
python run_bev_for_interaction_videos.py --id 2 --dataset_path /path/to/dataset_dir
python run_bev_for_interaction_videos.py --id 3 --dataset_path /path/to/dataset_dir
python run_bev_for_interaction_videos.py --id 4 --dataset_path /path/to/dataset_dir
3. Post process and merge results.
python post_process_for_interaction.py --dataset_path /path/to/dataset_dir
```

3. Go to `./Preprocess3Dpose` and Run following commands.
```
1. Preprocess extracted poses.
python interaction_preprocess.py --dataset_path /path/to/dataset_dir
2. Calc mean and variance.
python calc_mean_variance.py
```

After finishing these steps, `NTURGBD_multi`, which is used for interaction generation, is saved in `dataset_dir/preprocessed`.