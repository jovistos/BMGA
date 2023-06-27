import os,sys
import logging

import torchvision
torchvision.disable_beta_transforms_warning()

from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

sys.path.insert(1, '/home/jovis/Documents/WORK/Kaggle/BMGA/core/data')
from data import BeneData

import hydra
from omegaconf import DictConfig

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting the program")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES

    model_name = cfg.model_name

    processor = AutoProcessor.from_pretrained(model_name, is_vqa=False)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name)

    data_pipe = BeneData(
        processor=processor,
        labels_type=cfg.labels_type,
        labels_max_length=cfg.labels_max_length,
        frac_gen_train=cfg.frac_gen_train,
        classes_to_use=cfg.classes_to_use,
        use_augmentation=cfg.use_augmentation,
        use_synth=cfg.use_synth,
        synth_paths=cfg.synth_paths,
        synth_numbers=cfg.synth_numbers,
        logger=logger
    )

    train, valid = data_pipe.get_ds_splits()

    training_config = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        do_eval=True,
        evaluation_strategy=cfg.evaluation_strategy,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_accumulation_steps=cfg.eval_accumulation_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        logging_strategy=cfg.logging_strategy,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        no_cuda=cfg.no_cuda,
        seed=cfg.seed,
        bf16=cfg.bf16,
        load_best_model_at_end=cfg.load_best_model_at_end,
        torch_compile=cfg.torch_compile,
        optim=cfg.optim,
        report_to=cfg.report_to,
        remove_unused_columns=cfg.remove_unused_columns,
        dataloader_num_workers=cfg.dataloader_num_workers,
        prediction_loss_only=cfg.prediction_loss_only
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_config,
        data_collator=default_data_collator,
        train_dataset=train,
        eval_dataset=valid,
    )

    logger.info("Training started")
    train_result = trainer.train()
    logger.info("Training completed")


if __name__ == "__main__":
    main()
