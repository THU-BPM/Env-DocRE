api_key=$1

python icl/inference.py \
--api_key ${api_key} \
--model gpt-4-0125-preview \
--prompts_dir datasets/vanilla_prompts/1-shot/re-docred \
--predictions_dir icl/vanilla_predictions/gpt-4-0125-preview-1-shot/re-docred \
--doc_num 100

python icl/inference.py \
--api_key ${api_key} \
--model gpt-4-0125-preview \
--prompts_dir datasets/evr_prompts/1-shot/re-docred \
--predictions_dir icl/evr_predictions/gpt-4-0125-preview-1-shot/re-docred \
--doc_num 100

python icl/inference.py \
--api_key ${api_key} \
--model gpt-4-0125-preview \
--prompts_dir datasets/vanilla_prompts/3-shot/re-docred \
--predictions_dir icl/vanilla_predictions/gpt-4-0125-preview-3-shot/re-docred \
--doc_num 100

python icl/inference.py \
--api_key ${api_key} \
--model gpt-4-0125-preview \
--prompts_dir datasets/evr_prompts/3-shot/re-docred \
--predictions_dir icl/evr_predictions/gpt-4-0125-preview-3-shot/re-docred \
--doc_num 100