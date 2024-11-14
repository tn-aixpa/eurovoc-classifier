import argparse
import yaml
import os
import re
from transformers import AutoTokenizer
from skmultilearn.model_selection import IterativeStratification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
from torch import tensor, ones_like
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
import gzip
import pickle
import math
from copy import deepcopy
from datetime import datetime
from pagerange import PageRange
import shutil

from digitalhub_runtime_python import handler
seeds = []

src_basepath = "/shared"
file_basepath = "/files"

# The domains and microthesaurus labels are loaded from the json files
with open(f"{src_basepath}/src/config/domain_labels_position.json", "r") as fp:
    domain = json.load(fp)
with open(f"{src_basepath}/src/config/mt_labels_position.json", "r") as fp:
    microthesaurus = json.load(fp)
with open(f"{src_basepath}/src/config/mt_labels.json", "r", encoding="utf-8") as file:
    mt_labels = json.load(file)

def save_splits(data_path, seeds, X, masks, y, directory, mlb):
    """
    Save the splits of the dataset.
    
    :param X: List of inputs.
    :param masks: List of masks.
    :param y: List of labels.
    :param directory: Language directory.
    :param mlb: MultiLabelBinarizer object.
    """
    print(f"{datetime.now().replace(microsecond=0)} - Saving splits...")

    for seed in seeds:
        np.random.seed(int(seed))
        # Create two splits:test+dev and train
        stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.2, 0.8])
        train_idx, aux_idx = next(stratifier.split(X, y))
        train_X, train_mask, train_y = X[train_idx, :], masks[train_idx, :], y[train_idx, :]

        assert train_X.shape[0] == train_mask.shape[0] == train_y.shape[0]

        # Create two splits: test and dev
        stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.5, 0.5])
        dev_idx, test_idx = next(stratifier.split(X[aux_idx, :], y[aux_idx, :]))
        dev_X, dev_mask, dev_y = X[aux_idx, :][dev_idx, :], masks[aux_idx, :][dev_idx, :], y[aux_idx, :][dev_idx, :]
        test_X, test_mask, test_y = X[aux_idx, :][test_idx, :], masks[aux_idx, :][test_idx, :], y[aux_idx, :][test_idx, :]

        assert dev_X.shape[0] == dev_mask.shape[0] == dev_y.shape[0]
        assert test_X.shape[0] == test_mask.shape[0] == test_y.shape[0]

        to_print = f"{seed} - Splitted the documents in - train: {train_X.shape[0]}, dev: {dev_X.shape[0]}, test: {test_X.shape[0]}"
        print(to_print)

        with open(os.path.join(data_path, directory, "stats.txt"), "a+") as f:
            f.write(to_print + "\n")

        if not os.path.exists(os.path.join(data_path, directory, "output")):
            os.makedirs(os.path.join(data_path, directory, "output"))

        if not os.path.exists(os.path.join(data_path, directory, "output", f"split_{seed}")):
            os.makedirs(os.path.join(data_path, directory, "output", f"split_{seed}"))

        # Save the splits
        np.save(os.path.join(data_path, directory, "output", f"split_{seed}", "train_X.npy"), train_X)
        np.save(os.path.join(data_path, directory, "output", f"split_{seed}", "train_mask.npy"), train_mask)
        np.save(os.path.join(data_path, directory, "output", f"split_{seed}", "train_y.npy"), train_y)

        np.save(os.path.join(data_path, directory, "output", f"split_{seed}", "dev_X.npy"), dev_X)
        np.save(os.path.join(data_path, directory, "output", f"split_{seed}", "dev_mask.npy"), dev_mask)
        np.save(os.path.join(data_path, directory, "output", f"split_{seed}", "dev_y.npy"), dev_y)

        np.save(os.path.join(data_path, directory, "output", f"split_{seed}", "test_X.npy"), test_X)
        np.save(os.path.join(data_path, directory, "output", f"split_{seed}", "test_mask.npy"), test_mask)
        np.save(os.path.join(data_path, directory, "output", f"split_{seed}", "test_y.npy"), test_y)

        # Save the counts of each label, useful for weighted loss
        sample_labs = mlb.inverse_transform(train_y)
        labs_count = {"total_samples": len(sample_labs), "labels": {label: 0 for label in mlb.classes_}}

        for sample in sample_labs:
            for label in sample:
                labs_count["labels"][label] += 1
        
        with open(os.path.join(data_path, directory, "output", f"split_{seed}", "train_labs_count.json"), "w") as fp:
            json.dump(labs_count, fp)

        # Shuffle the splits using the random seed for reproducibility
        X, masks, y = shuffle(X, masks, y, random_state=int(seed))

def process_year(path, tokenizer_name, max_length, get_doc_ids, add_mt_do, title_only, add_title, summarized, limit_tokenizer):
    """
    Process a year of the dataset.

    :param path: Path to the data.
    :param tokenizer_name: Name of the tokenizer to use.
    :param args: Command line arguments.
    :return: List of inputs, masks and labels.
    """
    document_ct = 0
    big_document_ct = 0
    unk_ct = 0
    tokens_ct = 0

    inputs_ids = None
    labels = None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokenizer_kwargs = {"padding": "max_length", "truncation": True, "max_length": max_length}

    list_inputs = []
    list_labels = []
    list_masks = []

    with gzip.open(path, "rt", encoding="utf-8") as file:
        data = json.load(file)
        j = 1
        if get_doc_ids:
            # Only get the document ids, without processing the text. Useful to know which documents go in which split.
            for doc in data:
                print(f"{datetime.now().replace(microsecond=0)} - {j}/{len(data)}", end="\r")
                j += 1
                labels = data[doc]["eurovoc_classifiers"]

                inputs_ids = tensor(tokenizer.encode(doc, **tokenizer_kwargs))

                list_inputs.append(inputs_ids)
                list_labels.append(labels)
                list_masks.append(ones_like(inputs_ids))
        else:
            # Process the text
            for doc in data:
                print(f"{datetime.now().replace(microsecond=0)} - {j}/{len(data)}", end="\r")
                j += 1
                text = ""
                if add_mt_do:
                    # Add MT and DO labels
                    labels = set(data[doc]["eurovoc_classifiers"]) if "eurovoc_classifiers" in data[doc] else set(data[doc]["eurovoc"])
                    to_add = set()
                    for label in labels:
                        if label in mt_labels:
                            if mt_labels[label] in microthesaurus:
                                to_add.add(mt_labels[label] + "_mt")
                            if mt_labels[label][:2] in domain:
                                to_add.add(mt_labels[label][:2] + "_do")
                    
                    labels = list(labels.union(to_add))
                else:
                    labels = data[doc]["eurovoc_classifiers"] if "eurovoc_classifiers" in data[doc] else data[doc]["eurovoc"]

                if title_only:
                    text = data[doc]["title"]
                else:
                    if add_title:
                        text = data[doc]["title"] + " "
                    
                    if summarized:
                        full_text = data[doc]["full_text"]
                        phrase_importance = []
                        i = 0

                        for imp in data[doc]["importance"]:
                            if not math.isnan(imp):
                                phrase_importance.append((i, imp))
                            i += 1
                        
                        phrase_importance = sorted(phrase_importance, key=lambda x: x[1], reverse=True)

                        # First, we get the most important phrases until the maximum length is reached.
                        if len(" ".join([full_text[phrase[0]] for phrase in phrase_importance]).split()) > max_length:
                            backup = deepcopy(phrase_importance)
                            while len(" ".join([full_text[phrase[0]] for phrase in phrase_importance]).split()) > max_length:
                                phrase_importance = phrase_importance[:-1]
                            phrase_importance.append(backup[len(phrase_importance)])

                        # Then, we sort the phrases by their position in the document.
                        phrase_importance = sorted(phrase_importance, key=lambda x: x[0])
                        text += " ".join([full_text[phrase[0]] for phrase in phrase_importance])
                    else:
                        text += data[doc]["full_text"] if "full_text" in data[doc] else data[doc]["text"]
                
                text = re.sub(r'\r', '', text)
                
                if limit_tokenizer:
                    # Here, the text is cut to the maximum length before being tokenized,
                    # potentially speeding up the process for long documents.
                    inputs_ids = tensor(tokenizer.encode(text, **tokenizer_kwargs))
                else:
                    inputs_ids = tensor(tokenizer.encode(text))

                if not limit_tokenizer:
                    document_ct += 1

                    # We count the number of unknown tokens and the total number of tokens.
                    for token in inputs_ids[1: -1]:
                        if token == tokenizer.unk_token_id:
                            unk_ct += 1

                        tokens_ct += 1

                    # If the input is over the maximum length, we cut it and increment the count of big documents.
                    if len(inputs_ids) > max_length:
                        big_document_ct += 1
                        inputs_ids = inputs_ids[:max_length]

                list_inputs.append(inputs_ids)
                list_labels.append(labels)
                list_masks.append(ones_like(inputs_ids))
    
    del data, inputs_ids, labels, tokenizer

    # Just some stats to print and save later.
    if len(list_inputs) == 0:
        print("No documents found in the dataset.")
        to_print = ""
    else:
        if not limit_tokenizer and not get_doc_ids:
            to_print = f"Dataset stats: - total documents: {document_ct}, big documents: {big_document_ct}, ratio: {big_document_ct / document_ct * 100:.4f}%"
            to_print += f"\n               - total tokens: {tokens_ct}, unk tokens: {unk_ct}, ratio: {unk_ct / tokens_ct * 100:.4f}%"
            print(to_print)
        else:
            to_print = ""

    return list_inputs, list_masks, list_labels, to_print

def process_datasets(data_path, directory, tokenizer_name, seeds, years, max_length, get_doc_ids, add_mt_do, title_only, add_title, limit_tokenizer):
    """
    Process the datasets and save them in the specified directory.

    :param data_path: Path to the data.
    :param directory: Language directory.
    :param tokenizer_name: Name of the tokenizer to use.
    """

    list_inputs = []
    list_masks = []
    list_labels = []
    list_stats = []
    list_years = []

    # If no years are specified, process all the downloaded years depending on the arguments.
    summarized = False
    if years == "all":
        years = [year for year in os.listdir(os.path.join(data_path, directory))
                      if os.path.isfile(os.path.join(data_path, directory, year))
                      and year.endswith(".json.gz")]
    else:
        years = PageRange(years).pages
        files_in_directory = [file for file in os.listdir(os.path.join(data_path, directory))
                              if file.endswith(".json.gz")]

        are_any_summarized = ["sum" in file for file in files_in_directory]
        if any(are_any_summarized):
            sum_type = files_in_directory[are_any_summarized.index(True)].split("_", 1)[1]
            years = [str(year) + f"_{sum_type}" for year in years]
        else:
            years = [str(year) + ".json.gz" for year in years]
        
    years = sorted(years)
    
    # Test if the file is summarized or not
    with gzip.open(os.path.join(data_path, directory, years[0]), "rt", encoding="utf-8") as file:
        data = json.load(file)
        if "importance" in data[tuple(data.keys())[0]]:
            summarized = True
        del data

    print(f"Files to process: {', '.join(years)}\n")

    for year in years:
        print(f"Processing file: '{year}'...")
        year_inputs, year_masks, year_labels, year_stats = process_year(
            os.path.join(data_path, directory, year), 
            tokenizer_name, 
            max_length, get_doc_ids, add_mt_do, title_only, add_title, summarized, limit_tokenizer)
        
        list_inputs += year_inputs
        list_masks += year_masks
        list_labels += year_labels
        list_stats.append(year_stats)
        list_years.append(year)

    assert len(list_inputs) == len(list_masks) == len(list_labels)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(list_labels)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    X = pad_sequence(list_inputs, batch_first=True, padding_value=tokenizer.pad_token_id).numpy()
    masks = pad_sequence(list_masks, batch_first=True, padding_value=0).numpy()

    # Save the MultiLabelBinarizer.
    with open(os.path.join(data_path, directory, "mlb_encoder.pickle"), "wb") as pickle_fp:
        pickle.dump(mlb, pickle_fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    if not limit_tokenizer:
        with open(os.path.join(data_path, directory, "stats.txt"), "w") as stats_fp:
            for year, year_stats in zip(list_years, list_stats):
                stats_fp.write(f"Year: {year}\n{year_stats}\n\n")

    save_splits(data_path, seeds, X, masks, y, directory, mlb)

@handler()
def preprocess_data(project, 
                    classified_data, 
                    langs = "it",
                    years = "all",
                    seeds = "110",
                    add_title = False,
                    title_only = False,
                    max_length = 512,
                    limit_tokenizer = False,
                    add_mt_do = False,
                    get_doc_ids = False
                   ):
    """
    Load the configuration file and process the data.
    """

    data_path = f"{file_basepath}/data"
    try:
        shutil.rmtree(data_path)
    except:
        print("Error deleting files")                       
    classified_data.download(f"{file_basepath}/")
    
    with open(f"{src_basepath}/src/config/models.yml", "r") as fp:
        config = yaml.safe_load(fp)
    
    seeds = seeds.split(",")

    print(f"Tokenizers config:\n{format(config)}")
    
    for directory in os.listdir(data_path):
        # If we specified one or more languages, we only process those.
        if langs != "all" and directory not in langs.split(","):
            continue
        
        print(f"\nWorking on directory: {format(directory)}...")
        lang = directory
        print(f"Lang: '{lang}', Tokenizer: '{config[lang]}'")

        process_datasets(data_path, directory, config[lang], seeds, years, max_length, get_doc_ids, add_mt_do, title_only, add_title, limit_tokenizer)
        project.log_artifact(name=f"train_data_{lang}",
                                kind="artifact",
                                source=f"{data_path}/{directory}/output")

