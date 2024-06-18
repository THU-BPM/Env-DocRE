api_key=$1

python icl/inference.py \
--api_key ${api_key} \
--model gpt-3.5-turbo-0125 \
--prompts_dir datasets/vanilla_prompts/1-shot/re-docred \
--predictions_dir icl/vanilla_predictions/gpt-3.5-turbo-0125-1-shot/re-docred \
--doc_num 500

python icl/inference.py \
--api_key ${api_key} \
--model gpt-3.5-turbo-0125 \
--prompts_dir datasets/evr_prompts/1-shot/re-docred \
--predictions_dir icl/evr_predictions/gpt-3.5-turbo-0125-1-shot/re-docred \
--doc_num 500

python icl/inference.py \
--api_key ${api_key} \
--model gpt-3.5-turbo-0125 \
--prompts_dir datasets/vanilla_prompts/3-shot/re-docred \
--predictions_dir icl/vanilla_predictions/gpt-3.5-turbo-0125-3-shot/re-docred \
--doc_num 500

python icl/inference.py \
--api_key ${api_key} \
--model gpt-3.5-turbo-0125 \
--prompts_dir datasets/evr_prompts/3-shot/re-docred \
--predictions_dir icl/evr_predictions/gpt-3.5-turbo-0125-3-shot/re-docred \
--doc_num 500