from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import torch
import os
from transformers import __version__ as transformers_version
from packaging import version

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer & model (allow mismatched sizes to avoid warnings if needed)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt,
            ignore_mismatched_sizes=True
        ).to(device)

        # Data collator for sequence-to-sequence tasks
        seq2seq_data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model_pegasus
        )

        # Load dataset from disk
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Training arguments (disable pin_memory warning)
        trainer_args_kwargs = {
            "output_dir": self.config.root_dir,
            "num_train_epochs": 1,
            "warmup_steps": 500,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "weight_decay": 0.01,
            "logging_steps": 10,
            "eval_steps": 500,
            "save_steps": int(1e6),
            "gradient_accumulation_steps": 16,
            "dataloader_pin_memory": torch.cuda.is_available()
        }

        if version.parse(transformers_version) >= version.parse("3.1.0"):
            trainer_args_kwargs["evaluation_strategy"] = "steps"
        else:
            trainer_args_kwargs["evaluate_during_training"] = True

        trainer_args = TrainingArguments(**trainer_args_kwargs)