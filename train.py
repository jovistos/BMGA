
import os

from transformers import AutoProcessor, Pix2StructForConditionalGeneration

from transformers import  Seq2SeqTrainer,Seq2SeqTrainingArguments

from transformers import default_data_collator

import sys
sys.path.insert(1, '/home/jovis/Documents/WORK/Kaggle/BMGA/core/data')

from data import BeneData

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


model_name = "google/deplot"

processor = AutoProcessor.from_pretrained(model_name,is_vqa=False)
model = Pix2StructForConditionalGeneration.from_pretrained(model_name) 


data_pipe = BeneData(processor=processor,
                     labels_type="xy",
                     labels_max_length=730,
                     frac_gen_train=0.2,
                    classes_to_use=['line', 'horizontal_bar', 'scatter', 'vertical_bar', 'dot'],  
                     use_augmentation = True,
                     use_synth=False,
                     synth_paths= {"scatter":"/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/data/synth_data/scatter/v5",
                                "horizontal_bar":"/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/data/synth_data/horizontal_bar/v1_10000_5_40",
                                "vertical_bar":"/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/data/synth_data/vertical_bar/v1_1000_30_50",
                                "dot":"/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/data/synth_data/vertical_bar/v1_1000_30_50",
                                "line":"/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/data/synth_data/line/v3"},
                     synth_numbers = {"scatter":5000,
                                "horizontal_bar":5000,
                                "dot":0,
                                "vertical_bar":800,
                                "line":7000}
                    )
train, valid = data_pipe.get_ds_splits()


training_config = Seq2SeqTrainingArguments(
    output_dir="./output_l/hvdls_v1_l_v21",   
    do_eval=True,
    evaluation_strategy='steps',
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2, 
    gradient_accumulation_steps=16, 
    eval_accumulation_steps=None,
    eval_steps=60,
    save_steps=60,
    logging_steps=20,
    learning_rate=1e-5,
    num_train_epochs=10,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0,
    logging_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    no_cuda=False,
    seed=444,
    bf16=True,
    load_best_model_at_end=True,
    torch_compile=True,
    optim="adamw_torch",
    report_to=["tensorboard"],
    remove_unused_columns=False,
    dataloader_num_workers=1,
    prediction_loss_only=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_config,
    data_collator=default_data_collator, 
    train_dataset=train,    
    eval_dataset=valid, 
)

train_result = trainer.train()
trainer.save_model()  
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()






