devices=$1
log_file_name=$2

CUDA_VISIBLE_DEVICES=${devices} nohup python -u evrt/main.py \
--data_dir datasets/env-re-docred \
--transformer_type roberta \
--model_name_or_path roberta-large \
--train_file train_revised.json \
--evrt_file evrt.json \
--dev_file dev_revised_env.json \
--test_file test_revised_env.json \
--rel2id_file rel2id.json \
--output_dir ${log_file_name} \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 2 \
--num_labels 4 \
--learning_rate 3e-5 \
--classifier_lr 1e-4 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--clsloss_shift 0.04 \
--clsloss_reg \
--evrt \
--pcrloss_weight 0.2 \
--rcrloss_weight 0.2 \
--num_train_epochs 8.0 \
--seed 42 \
--num_class 97 \
> ${log_file_name}.log 2>&1 &