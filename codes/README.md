## Dataset and Pretrained Weights
<details>
<summary>Prepare pretrained weights and data</summary>

If you want to train models, please preprocess dataset in `../preprocess` at first.  
`data/NTURGBD_multi` already includes train/val/test split files and `test_active_anns.json`, which is small set annotations regarding an actor and a receiver.
Please move all files in `NTURGBD_multi`, which is prepared in `../preprocess`, to `data/NTURGBD_multi`.  

Please download [pretrained weights](https://drive.google.com/drive/folders/1PcZf8T8g24olUER77WH4mA88Kc4_rhHI?usp=drive_link).  
If you want to train an interaction generation model using a pretrained single motion generation model, please download MotionDiffuse model from [here](https://github.com/mingyuan-zhang/MotionDiffuse/blob/main/text2motion/install.md).   
The whole directory is as follows.

```
codes
└───checkpoints
│   └───ntu_mul
│   │   └───interaction
│   │   └───interaction_pretrained
│   │   └───pit
│   │   └───eval_model
│   │   └───consistency_eval_model
│   └───t2m # pretrained model
│       └───t2m_motiondiffuse 
└───data/NTURGBD_multi
│   └───new_joint_vecs
│   └───newjoints
│   └───test_active_anns.json
│   └───pseudo_labels.json
│   └───...
└───...
```

</details>


## Installation
<details>
<summary>Setup</summary>

```
conda create -n motioninteraction python=3.7 -y
source activate motioninteraction

# install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# install mmcv
pip install "mmcv-full>=1.3.17,<=1.5.3" -f https://download.openmmlab.com/mmcv/dist/11.3/1.9.0/index.html

pip install -r requirements.txt
```

</details>

## Training interaction generation model

(1-1)  
The NTURGBD dataset contains only interaction labels and does not include annotations for individual roles. We introduced PIT to train a label-based role-aware interaction generation model at first.  

(1-2)  
The obtained roles can be identified using a small amount of annotations regarding an actor and a receiver.  
After identifying the roles, you can provide pseudo labels regarding which motion is an actor or a receiver for all training data.  

(1-3)  
By using the above pseudo labels for individual roles, you can train a model that can generate role-aware interactions from texts.  


<details>
<summary>(1-1) Training label-based role-aware interaction generation model (PIT)</summary>

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -u tools/train.py \
    --name pit \
    --batch_size 160 \
    --times 30 \
    --num_epochs 50 \
    --dataset_name ntu_mul \
    --multi \
    --cap_id
```

</details>

<details>
<summary>(1-2) Providing pseudo-labels for each role</summary>

Identify obtained roles by using `data/NTURGBD_multi/test_active_anns.json`, which is a small amount of annotations regarding an actor and a receiver.

```
CUDA_VISIBLE_DEVICES=0 python -u tools/label_data.py \
    --opt_path checkpoints/ntu_mul/pit/opt.txt \
    --label_path data/NTURGBD_multi/test_active_anns.json \
    --which_epoch latest \
    --label_model
```

Provide pseudo labels regarding roles for all training data by the following command.

```
python -u tools/label_data.py \
    --opt_path checkpoints/ntu_mul/pit/opt.txt \
    --which_epoch latest \
    --save_label
```

</details>

<details>
<summary>(1-3) Training models for role-aware interaction generation from texts</summary>

If you want to use a pretrained single motion generation model, please add flag `--pretrained`.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u tools/train.py \
    --name interaction_pretrained \
    --batch_size 120 \
    --times 200 \
    --num_epochs 50 \
    --dataset_name ntu_mul \
    --multi \
    --label_path ./data/NTURGBD_multi/pseudo_labels.json \
    --port 12345
```

</details>

## Training evaluation model

<details>
<summary>Training models for evaluation</summary>

Train an interaction recognition model to evaluate Accuracy, FID, and so on.
```
CUDA_VISIBLE_DEVICES=0 python -u tools/train_evaluation_model.py \
    --name eval_model \
    --batch_size 80 \
    --num_epochs 100 \
    --dataset_name ntu_mul \
    --multi 
```

Train a mutual consistency evaluation model.
```
CUDA_VISIBLE_DEVICES=0 python -u tools/train_consistency_evaluation_model.py \
    --name consistency_eval_model \
    --batch_size 80 \
    --num_epochs 100 \
    --dataset_name ntu_mul \
    --multi 
```

</details>

## Evaluation

<details>
<summary>Evaluating generated results</summary>

```
CUDA_VISIBLE_DEVICES=0 python -u tools/evaluation.py \
    --opt_path checkpoints/ntu_mul/interaction/opt.txt \
    --gpu_id 0 \
    --model_name latest \
    --split_file test_sub.txt \
    --file_id 0
```

</details>

## Visualization

<details>
<summary>Visualizing generated motions</summary>

Run the following command.  
--result_path specifies the save directory.

```
## inference for role-aware interaction generation from texts
CUDA_VISIBLE_DEVICES=0 python -u tools/visualization.py \
    --opt_path checkpoints/ntu_mul/interaction_pretrained/opt.txt \
    --text_category 0\
    --result_path vis_data/gen_interaction \
    --which_epoch latest \
    --gpu_id 0 \
    --interaction    \
    --motion_length 60 

## inference for label-based role-aware interaction generation model (PIT)
CUDA_VISIBLE_DEVICES=0 python -u tools/visualization.py \
    --opt_path checkpoints/ntu_mul/pit/opt.txt \
    --text_category 0\
    --result_path vis_data/pit \
    --which_epoch latest \
    --gpu_id 0 \
    --interaction    \
    --motion_length 60  \
    --cap_id
```

If you want to render the generated sequence with the SMPL human model, please go to `./joints2smpl` and visualize it by using the npy file saved in --result_path.

</details>

## Acknowledgement
Our codes are based on [MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse/tree/main).