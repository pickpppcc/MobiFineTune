"""This script samples K examples randomly without replacement from the original data."""

import argparse
import os
import numpy as np
import pandas as pd
from pandas import DataFrame

def get_label(task, line):
    if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
        # GLUE style
        line = line.strip().split('\t')
        if task == 'CoLA':
            return line[1]
        elif task == 'MNLI':
            return line[-1]
        elif task == 'MRPC':
            return line[0]
        elif task == 'QNLI':
            return line[-1]
        elif task == 'QQP':
            return line[-1]
        elif task == 'RTE':
            return line[-1]
        elif task == 'SNLI':
            return line[-1]
        elif task == 'SST-2':
            return line[-1]
        elif task == 'STS-B':
            return 0 if float(line[-1]) < 2.5 else 1
        elif task == 'WNLI':
            return line[-1]
        else:
            raise NotImplementedError
    else:
        return line[0]

def load_datasets(data_dir, tasks):
    datasets = {}
    for task in tasks:
        if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
            # GLUE style (tsv)
            dataset = {}
            dirname = os.path.join(data_dir, task)
            if task == "MNLI":
                splits = ["train", "dev_matched", "dev_mismatched"]
            else:
                splits = ["train", "dev"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.tsv")
                with open(filename, "r") as f:
                    lines = f.readlines()
                dataset[split] = lines
            datasets[task] = dataset
        else:
            # Other datasets (csv)
            dataset = {}
            dirname = os.path.join(data_dir, task)
            splits = ["train", "test"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.csv")
                dataset[split] = pd.read_csv(filename, header=None)
            datasets[task] = dataset
    return datasets

def split_header(task, lines):
    """
    Returns if the task file has a header or not. Only for GLUE tasks.
    """
    if task in ["CoLA"]:
        return [], lines
    elif task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI"]:
        return lines[0:1], lines[1:]
    else:
        raise ValueError("Unknown GLUE task.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, nargs="+",
        default=['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA', 'MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE'],
        help="Task names")
    parser.add_argument("--seed", type=int, nargs="+",
        default=[100, 13, 21, 42, 87],
        help="Random seeds")

    parser.add_argument("--data_dir", type=str, default="data/original", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output path")
    parser.add_argument("--mode", type=str, default='k-shot', choices=['k-shot', 'k-shot-10x', 'k-shot-1k-test'], help="k-shot or k-shot-10x (10x dev set)")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.mode)

    # k = args.k
    # print("K =", k)
    datasets = load_datasets(args.data_dir, args.task)

    for seed in args.seed:
        print("Seed = %d" % (seed))
        for task, dataset in datasets.items():
            # Set random seed
            np.random.seed(seed)

            # Shuffle the training set
            print("| Task = %s" % (task))
            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                # GLUE style
                train_header, train_lines = split_header(task, dataset["train"])
                np.random.shuffle(train_lines)
            else:
                # Other datasets
                train_lines = dataset['train'].values.tolist()
                np.random.shuffle(train_lines)
            
            # Set up dir
            task_dir = os.path.join(args.output_dir, task)
            setting_dir = os.path.join(task_dir, f"{seed}")
            os.makedirs(setting_dir, exist_ok=True)

            # Write test splits
            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                # GLUE style
                # Use the original development set as the test set (the original test sets are not publicly available)
                for split, lines in dataset.items():
                    if split.startswith("train"):
                        continue
                    split = split.replace('dev', 'test')

                    test_header, test_lines = split_header(task, lines)
                    if '1k-test' in args.mode and len(test_lines) > 1000:
                        np.random.seed(42)
                        np.random.shuffle(test_lines)
                        test_lines = test_lines[:1000]
                    with open(os.path.join(setting_dir, f"{split}.tsv"), "w") as f:
                        for line in test_header:
                            f.write(line)
                        for line in test_lines:
                            f.write(line)
            else:
                # Other datasets
                # Use the original test sets
                test_dataset = dataset['test']
                if '1k-test' in args.mode and len(test_dataset.index) > 1000:
                    test_dataset = test_dataset.sample(n=1000, random_state=42)
                test_dataset.to_csv(os.path.join(setting_dir, 'test.csv'), header=False, index=False)

            # Get label list for balanced sampling
            label_list = {}
            for line in train_lines:
                label = get_label(task, line)
                if label not in label_list:
                    label_list[label] = [line]
                else:
                    label_list[label].append(line)

            num_train=int(len(train_lines)*0.8)

            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                with open(os.path.join(setting_dir, "train.tsv"), "w") as f:
                    for line in train_header:
                        f.write(line)
                    for label in label_list:
                        num=int(len(label_list[label])*0.8)
                        for line in label_list[label][:num]:
                            f.write(line)
                name = "dev.tsv"
                if task == 'MNLI':
                    name = "dev_matched.tsv"
                with open(os.path.join(setting_dir, name), "w") as f:
                    for line in train_header:
                        f.write(line)
                    for label in label_list:
                        num=int(len(label_list[label])*0.8)
                        for line in label_list[label][num:]:
                            f.write(line)
            else:
                new_train = []
                for label in label_list:
                    num=int(len(label_list[label])*0.8)
                    for line in label_list[label][:num]:
                        new_train.append(line)
                new_train = DataFrame(new_train)
                new_train.to_csv(os.path.join(setting_dir, 'train.csv'), header=False, index=False)

                new_dev = []
                for label in label_list:
                    num=int(len(label_list[label])*0.8)
                    for line in label_list[label][num:]:
                        new_dev.append(line)
                new_dev = DataFrame(new_dev)
                new_dev.to_csv(os.path.join(setting_dir, 'dev.csv'), header=False, index=False)


if __name__ == "__main__":
    main()
