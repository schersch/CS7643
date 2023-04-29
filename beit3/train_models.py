import subprocess
import os

#---------------------------------
# Begin original code contribution
#---------------------------------

models = [
    {"size": "base", "seed": 42, "lr": "7e-4", "batch_size": 32, "data": "unaug"}, 
    {"size": "base", "seed": 42, "lr": "7e-4", "batch_size": 32, "data": "aug"}, 
    {"size": "large", "seed": 47, "lr": "3e-4", "batch_size": 16, "data": "unaug"}, 
    {"size": "large", "seed": 48, "lr": "3e-4", "batch_size": 46, "data": "unaug"}, 
    {"size": "large", "seed": 49, "lr": "3e-4", "batch_size": 16, "data": "unaug"}, 
    {"size": "large", "seed": 50, "lr": "3e-4", "batch_size": 16, "data": "unaug"}, 
    {"size": "large", "seed": 51, "lr": "3e-4", "batch_size": 16, "data": "unaug"}, 
    {"size": "large", "seed": 47, "lr": "3e-4", "batch_size": 16, "data": "aug"}, 
    {"size": "large", "seed": 48, "lr": "3e-4", "batch_size": 16, "data": "aug"}, 
    {"size": "large", "seed": 49, "lr": "3e-4", "batch_size": 16, "data": "aug"}, 
    {"size": "large", "seed": 50, "lr": "3e-4", "batch_size": 16, "data": "aug"}, 
    {"size": "large", "seed": 51, "lr": "3e-4", "batch_size": 16, "data": "aug"}
]


def train_models(model_list):
    for model_params in model_list:
        train_one_model(**model_params)


def train_one_model(size, batch_size, lr, seed, data):
        train_cmd = f"python -m torch.distributed.launch --nproc_per_node=1 run_beit3_finetuning.py \
            --model beit3_{size}_patch16_224 \
            --task hateful_memes \
            --batch_size {batch_size} \
            --layer_decay 0.85 \
            --lr {lr} \
            --epochs 20 \
            --warmup_epochs 5 \
            --drop_path 0.2 \
            --sentencepiece_model beit3.spm \
            --finetune models/pretrained/beit3_{size}_patch16_224.pth \
            --data_path hm_data \
            --data {data} \
            --output_dir models/finetuned/beit3_{seed}_{size}_{data} \
            --log_dir logs \
            --weight_decay 0.2 \
            --seed {seed} \
            --enable_deepspeed \
            --save_ckpt_freq 20 \
            --checkpoint_activations"
        
        subprocess.run(train_cmd, shell=True)


def eval_models(model_list):
    for model_params in model_list:
        eval_one_model(**model_params)


def eval_one_model(size, batch_size, lr, seed, data):
    eval_cmd = f"python -m torch.distributed.launch --nproc_per_node=1 run_beit3_finetuning.py \
        --model beit3_{size}_patch16_224 \
        --task hateful_memes \
        --batch_size {batch_size} \
        --sentencepiece_model beit3.spm \
        --finetune models/finetuned/beit3_{seed}_{size}_{data}/checkpoint-best/mp_rank_00_model_states.pt \
        --data_path hm_data \
        --data {data} \
        --eval \
        --dist_eval"

    subprocess.run(eval_cmd, shell=True)
    os.rename("results/preds.json", f"results/beit3_{seed}_{size}_{data}_preds.json")



if __name__ == "__main__":
    train_models(models)
    eval_models(models)

#-------------------------------
# End original code contribution
#-------------------------------