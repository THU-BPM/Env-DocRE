import argparse
import math
import os

import numpy as np
import torch
import ujson as json
from apex import amp
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, AdamW

from evaluation import to_official, official_evaluate, official_evaluate_benchmark
from losses import PcrLoss, RcrLoss
from model import DocREModel
from prepro import read_docred
from utils import set_seed, collate_fn


def train(args, model, optimizer, train_features, dev_features, test_features, rel2id, id2rel, tokenizer, suffix):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score, best_epoch = -1, -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=False)
        train_iterator = range(int(num_epoch))
        total_steps = int(math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * num_epoch)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        if args.evrt:
            pcr_loss_fnt = PcrLoss()
            rcr_loss_fnt = RcrLoss()
            if os.path.exists(os.path.join(args.data_dir, args.evrt_file + suffix)):
                crt_features = torch.load(os.path.join(args.data_dir, args.evrt_file + suffix))
            else:
                crt_file = os.path.join(args.data_dir, args.evrt_file)
                crt_features = read_docred(crt_file, rel2id, tokenizer, max_seq_length=args.max_seq_length)
                torch.save(crt_features, os.path.join(args.data_dir, args.evrt_file + suffix))
            crt_titles_to_features = {feature["title"]: feature for feature in crt_features}
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps

                if args.evrt:
                    crt_batch = collate_fn([crt_titles_to_features[title] for title in batch[5]])
                    crt_inputs = {'input_ids': crt_batch[0].to(args.device),
                                  'attention_mask': crt_batch[1].to(args.device),
                                  'labels': crt_batch[2],
                                  'entity_pos': crt_batch[3],
                                  'hts': crt_batch[4],
                                  }
                    crt_outputs = model(**crt_inputs)
                    loss += crt_outputs[0] / args.gradient_accumulation_steps

                    pcr_loss = pcr_loss_fnt(outputs[2], crt_outputs[2])
                    loss += args.pcrloss_weight * pcr_loss / args.gradient_accumulation_steps

                    rcr_loss = rcr_loss_fnt(outputs[3], crt_outputs[3])
                    loss += args.rcrloss_weight * rcr_loss / args.gradient_accumulation_steps

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                if step == len(train_dataloader) - 1 or (
                        args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and (
                        step + 1) % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = evaluate(args, model, dev_features, rel2id, id2rel, tag="dev")
                    print('Dev results of epoch {}:'.format(epoch + 1))
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        best_epoch = epoch + 1
                        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pth"))
                    if direct_test(args.data_dir):
                        test_score, test_output = evaluate(args, model, test_features, rel2id, id2rel, tag="test")
                        print('Test results of epoch {}:'.format(epoch + 1))
                        print(test_output)
                    else:
                        pred = report(args, model, test_features, rel2id, id2rel)
                        with open(os.path.join(args.output_dir, "result_epoch={}.json".format(epoch + 1)), "w") as fh:
                            json.dump(pred, fh)
        print('Best epoch:', best_epoch)
        print('Best dev F1 + Ign F1:', best_score)
        return num_steps

    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, rel2id, id2rel, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features, id2rel)
    if tag == 'dev':
        result = official_evaluate_benchmark(ans, args.data_dir, args.train_file, args.dev_file, rel2id)
    else:
        result = official_evaluate_benchmark(ans, args.data_dir, args.train_file, args.test_file, rel2id)
    output = {
        tag + "_re_f1": result[0] * 100,
        tag + "_evi_f1": result[1] * 100,
        tag + "_re_f1_ignore_train_annotated": result[2] * 100,
        tag + "_re_f1_ignore_train": result[3] * 100,
        tag + "_re_p": result[4] * 100,
        tag + "_re_r": result[5] * 100,
        tag + "_re_f1_freq": result[6] * 100,
        tag + "_re_f1_long_tail": result[7] * 100,
        tag + "_re_f1_intra": result[8] * 100,
        tag + "_re_f1_inter": result[9] * 100,
        tag + "_re_p_freq": result[10] * 100,
        tag + "_re_r_freq": result[11] * 100,
        tag + "_re_p_long_tail": result[12] * 100,
        tag + "_re_r_long_tail": result[13] * 100
    }
    return result[0] * 100 + result[2] * 100, output


def report(args, model, features, rel2id, id2rel):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features, id2rel)
    return preds


def direct_test(data_dir):
    dataset_list = ['re-docred']
    for dataset in dataset_list:
        if dataset in data_dir:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="datasets/env-docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--evrt_file", default="evrt.json", type=str)
    parser.add_argument("--dev_file", default="dev_env.json", type=str)
    parser.add_argument("--test_file", default="test_env.json", type=str)
    parser.add_argument("--rel2id_file", default="rel2id.json", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--output_dir", default="output", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of update steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--classifier_lr", default=1e-4, type=float,
                        help="The initial learning rate for classifier.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--clsloss_shift", default=0.05, type=float,
                        help="Probability shift in ClsLoss.")
    parser.add_argument("--clsloss_reg", action="store_true",
                        help="Add regularization term for ClsLoss.")
    parser.add_argument("--evrt", action="store_true",
                        help="Enable entity variance robust training.")
    parser.add_argument("--pcrloss_weight", default=1.0, type=float,
                        help="Weight of PcrLoss.")
    parser.add_argument("--rcrloss_weight", default=1.0, type=float,
                        help="Weight of RcrLoss.")
    parser.add_argument("--num_train_epochs", default=8.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--only_test", action="store_true",
                        help="Only test on the development and test sets.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    suffix = '.{}.pt'.format(args.model_name_or_path.split('/')[-1].strip())
    read = read_docred
    with open(os.path.join(args.data_dir, args.rel2id_file), 'r', encoding='utf-8') as f:
        rel2id = json.load(f)
    id2rel = {value: key for key, value in rel2id.items()}

    if os.path.exists(os.path.join(args.data_dir, args.train_file + suffix)):
        train_features = torch.load(os.path.join(args.data_dir, args.train_file + suffix))
        print('Loaded train features')
    else:
        train_file = os.path.join(args.data_dir, args.train_file)
        train_features = read(train_file, rel2id, tokenizer, max_seq_length=args.max_seq_length)
        torch.save(train_features, os.path.join(args.data_dir, args.train_file + suffix))
        print('Created and saved new train features')
    if os.path.exists(os.path.join(args.data_dir, args.dev_file + suffix)):
        dev_features = torch.load(os.path.join(args.data_dir, args.dev_file + suffix))
        print('Loaded dev features')
    else:
        dev_file = os.path.join(args.data_dir, args.dev_file)
        dev_features = read(dev_file, rel2id, tokenizer, max_seq_length=args.max_seq_length)
        torch.save(dev_features, os.path.join(args.data_dir, args.dev_file + suffix))
        print('Created and saved new dev features')
    if os.path.exists(os.path.join(args.data_dir, args.test_file + suffix)):
        test_features = torch.load(os.path.join(args.data_dir, args.test_file + suffix))
        print('Loaded test features')
    else:
        test_file = os.path.join(args.data_dir, args.test_file)
        test_features = read(test_file, rel2id, tokenizer, max_seq_length=args.max_seq_length)
        torch.save(test_features, os.path.join(args.data_dir, args.test_file + suffix))
        print('Created and saved new test features')

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(args, config, model)
    model.to(device)

    if args.only_test:
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        if args.load_path != "":
            model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, rel2id, id2rel, tag="dev")
        print('Dev results:')
        print(dev_output)
        if direct_test(args.data_dir):
            test_score, test_output = evaluate(args, model, test_features, rel2id, id2rel, tag="test")
            print('Test results:')
            print(test_output)
        else:
            pred = report(args, model, test_features, rel2id, id2rel)
            with open(os.path.join(args.output_dir, "result.json"), "w") as fh:
                json.dump(pred, fh)
    else:
        new_layer = ["extractor", "bilinear"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)],
             "lr": args.classifier_lr},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        if args.load_path != "":
            model.load_state_dict(torch.load(args.load_path))
        train(args, model, optimizer, train_features, dev_features, test_features, rel2id, id2rel, tokenizer, suffix)


if __name__ == "__main__":
    main()
