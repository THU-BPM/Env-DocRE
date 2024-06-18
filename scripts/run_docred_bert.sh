devices=$1
log_file_name=$2

CUDA_VISIBLE_DEVICES=${devices} nohup python -u evrt/main.py \
--data_dir datasets/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--train_file train_annotated.json \
--evrt_file evrt.json \
--dev_file dev.json \
--test_file test.json \
--rel2id_file rel2id.json \
--output_dir ${log_file_name} \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--learning_rate 5e-5 \
--classifier_lr 1e-4 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--clsloss_shift 0.05 \
--clsloss_reg \
--evrt \
--pcrloss_weight 1.0 \
--rcrloss_weight 1.0 \
--num_train_epochs 8.0 \
--seed 42 \
--num_class 97 \
> ${log_file_name}.log 2>&1 &