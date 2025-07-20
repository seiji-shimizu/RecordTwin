import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

logger = logging.getLogger(__name__)

label_list_2010 = ['O', 'B-treatment', 'I-treatment', 'B-problem', 'I-problem', 'B-test', 'I-test']
label_list_2012 = ['B-OCCURRENCE', 'O', 'B-TREATMENT', 'I-OCCURRENCE', 'B-PROBLEM', 'I-PROBLEM', 'B-CLINICAL_DEPT', 'I-CLINICAL_DEPT', 'B-TEST', 'I-TEST', 'B-EVIDENTIAL', 'I-TREATMENT', 'I-EVIDENTIAL']


#######################################################################################################################################
# Args

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    label_all_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one token in the sequence or to put the label for the whole word."
        })

@dataclass
class TokenizerArguments:
    max_length: Optional[int] = field(
        default=4096, metadata={"help": "The maximum length of the tokenized input sequences."}
    )
    padding: Optional[str] = field(
        default="max_length", metadata={"help": "Padding strategy. Options: 'longest', 'max_length', 'do_not_pad'"}
    )

#######################################################################################################################################



def get_label_list(label_list):
        unique_labels = set(label_list)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TokenizerArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, tokenizer_args, training_args = parser.parse_args_into_dataclasses()
        # check num epochs
        if training_args.num_train_epochs is None:
            raise ValueError("num_train_epochs must be specified")
        else:
            print(f"num_train_epochs: {training_args.num_train_epochs}")
    #######################################################################################################################################
    # Logging
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    #######################################################################################################################################
    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    #######################################################################################################################################
    task_name_year = data_args.train_file.split('/')[-2]
    if task_name_year == 'i2b2_2012':
            num_labels = len(label_list_2012)
            label_list = get_label_list(label_list_2012)
            label_to_id = {label: i for i, label in enumerate(label_list)}
    
    #######################################################################################################################################
    # Load pretrained model and tokenizer
    
    # Tokenizer
    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    
    tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=True,
            revision=model_args.model_revision,
            add_prefix_space=True
        )
    
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )


    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label={i: l for l, i in label_to_id.items()},
        finetuning_task=data_args.task_name,
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    
    #######################################################################################################################################
    # Tokenization and Labeling
    # Preprocessing the dataset
    
    max_len = tokenizer_args.max_length
    padding = tokenizer_args.padding

    def map_fnc(examples):
        tokenized_inputs = tokenizer(examples['text'], padding=padding, max_length=max_len, truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    ####################################################################################################################################################################################
    # Dataset
    
    def return_datasets(data_files): 
        raw_datasets = load_dataset('json', data_files=data_files)
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.map(
                map_fnc,
                batched=True,
                desc="Running tokenizer on train dataset",
                load_from_cache_file=False
            )
        eval_dataset = raw_datasets["dev"]
        eval_dataset = eval_dataset.map(
                map_fnc,
                batched=True,
                desc="Running tokenizer on eval dataset",
                load_from_cache_file=False
            )
        test_dataset = raw_datasets["test"]
        test_dataset = test_dataset.map(
                map_fnc,
                batched=True,
                desc="Running tokenizer on test dataset",
                load_from_cache_file=False
            )
        return train_dataset, eval_dataset, test_dataset
    
    #######################################################################################################################################
    # Metrics
    metric = load_metric("seqeval")
    
    def compute_metrics(p):
        # logits, label =
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
        
    
    #######################################################################################################################################
    
    # Training, Validation, Testing
    # Initialize our Trainer
    # Take Dataset and Metrics as arguments
    train_dataset, eval_dataset, predict_dataset = return_datasets({'train': data_args.train_file, 'dev': data_args.validation_file, 'test': data_args.test_file})

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    #######################################################################################################################################
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict / Evaluation
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)]]
        # and convert predictions to list of labels
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")
        
        # Save true labels
        output_label_file = os.path.join(training_args.output_dir, "labels.txt")
        if trainer.is_world_process_zero():
            with open(output_label_file, "w") as writer:
                for label in true_labels:
                    writer.write(" ".join(label) + "\n")

if __name__ == "__main__":
    main()
