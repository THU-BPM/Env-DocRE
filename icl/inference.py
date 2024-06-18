import argparse
import os
import time

from openai import OpenAI
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--prompts_dir", type=str, default="datasets/vanilla_prompts/1-shot/re-docred")
    parser.add_argument("--predictions_dir", type=str,
                        default="icl/vanilla_predictions/gpt-3.5-turbo-0125-1-shot/re-docred")
    parser.add_argument("--doc_num", type=int, default=500)
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    if not os.path.exists(args.predictions_dir):
        os.makedirs(args.predictions_dir)

    prompt_files = [str(i) for i in range(args.doc_num)]
    for prompt_file in tqdm(prompt_files, desc="Doc"):
        prompt_file_path = os.path.join(args.prompts_dir, prompt_file)
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=args.model,
                max_tokens=1024,
                temperature=0
            )
            predictions = completion.choices[0].message.content.strip()
            with open(os.path.join(args.predictions_dir, prompt_file), "w", encoding="utf-8") as f:
                f.write(predictions)
        except Exception as e:
            print("API ERROR at {}:".format(prompt_file), e)
            time.sleep(60)
            continue
        time.sleep(1)


if __name__ == "__main__":
    main()
