# see https://www.kaggle.com/code/joviis/deplot-infer-v24 
# for weights and execution


import re
from pathlib import Path
from typing import List
from functools import partial


import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import Image as ds_img
from tqdm.auto import tqdm
from torchvision.transforms import Compose, ColorJitter,GaussianBlur,Resize
from transformers import AutoProcessor,Pix2StructForConditionalGeneration,  Pix2StructProcessor, AutoConfig



val_transform = Resize(size=500, antialias=True)



def inference_ds(model,ds):
    model.eval()
    device = torch.device("cuda:0")
    model.to(device)

    data_loader = DataLoader(
        ds, batch_size=4, shuffle=False
    )


    all_generations = []
    for batch in tqdm(data_loader):
        flattened_patches = batch["flattened_patches"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        batch_size = flattened_patches.shape[0]

        try:
            generated_ids = model.generate(
                flattened_patches=flattened_patches, 
                attention_mask=attention_mask, 
                max_length=512,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True)


            all_generations.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
        except:
            all_generations.extend([""]*batch_size)
    return all_generations


def tokenize_image(examples):
    
    tokenized_examples = processor(images = examples["image_path"], 
                                   is_vqa=False,
                       add_special_tokens=True,
                    return_tensors="pt")

    examples["flattened_patches"] = tokenized_examples["flattened_patches"]
    examples["attention_mask"] = tokenized_examples["attention_mask"]
    examples["image_path"]=tokenized_examples["attention_mask"]
    
    return  examples 


def resize_tokenize(examples):
    examples["image_path"] = [val_transform(image.convert("RGB")) for image in examples["image_path"]]
    
    return tokenize_image(examples)



def tokenize(examples):
    
    return tokenize_image(examples)



def convert_to_ds(dff,resize=False):
    
    ds = Dataset.from_pandas(dff.copy().reset_index(drop=True))

    ds = ds.cast_column("image_path", ds_img())
    
    if resize == True:
        ds.set_transform(resize_tokenize)
    else:
        ds.set_transform(tokenize)
        
#     ds.set_transform(tokenize_image,resize=resize)
    
    return ds



def xy_to_x(xy,dim = 0, include_count=False):
    try:
        # x = [y.split("^")[dim] for y in xy.split(";")]
        if include_count:
            x = [y.split("^")[dim] for y in xy.split("**")[1].split(";")]
        else:
            x = [y.split("^")[dim] for y in xy.split(";")]
        
    except:
        x = []
    return x



def infer_part(df,ch_types,model_main_path,include_count=True,resize=False):
    ds = convert_to_ds(df[df.pred_class.isin(ch_types)],resize=resize)
    model_main = Pix2StructForConditionalGeneration.from_pretrained(model_main_path)  

    all_generations_main = inference_ds(model_main,ds)
    df["pred_xy"][df.pred_class.isin(ch_types)] = all_generations_main

    del model_main
    torch.cuda.empty_cache()

    df["pred_x"][df.pred_class.isin(ch_types)] = df["pred_xy"][df.pred_class.isin(ch_types)].apply(lambda x:xy_to_x(x,dim=0, include_count=include_count))
    df["pred_y"][df.pred_class.isin(ch_types)] = df["pred_xy"][df.pred_class.isin(ch_types)].apply(lambda x:xy_to_x(x,dim=1, include_count=include_count))
    
    return df


class CFG:
    batch_size = 4
    image_path = "/kaggle/input/benetech-making-graphs-accessible/test/images"




model_path="../input/bmga-dp-v3-a"
# model_name = "/kaggle/input/google-deplot-model"
proc_path = "/kaggle/input/deplot-processor/deplot"
processor =  Pix2StructProcessor.from_pretrained(proc_path,is_vqa=False) #,force_download=True


image_dir = Path(CFG.image_path)
images = list(image_dir.glob("*.jpg"))

df = pd.DataFrame.from_dict(
    {"image_path": [str(x) for x in images], "id": [x.stem for x in images]}
)
ds = convert_to_ds(df,resize=True)


model_class_path="/kaggle/input/cls-v3/checkpoint-17000/"
model_class = Pix2StructForConditionalGeneration.from_pretrained(model_class_path)

all_generations_class = inference_ds(model_class,ds)
df["pred_class"] = all_generations_class

del model_class
torch.cuda.empty_cache()



df["pred_xy"]="None"
df["pred_x"]="None"
df["pred_y"]="None"

df = infer_part(df,["scatter"],"/kaggle/input/hvdls-v1-s-v13-gen/checkpoint-500",resize=True)
df = infer_part(df,["line"],"/kaggle/input/hvdls-v1-l-v12/checkpoint-680",resize=True)
df = infer_part(df,["dot"],"/kaggle/input/hvdls-v1-dot/checkpoint-36300/",resize=True)
df = infer_part(df,["vertical_bar"],"/kaggle/input/hvdls-v1-hv-v2/checkpoint-1140",resize=True)
df = infer_part(df,["horizontal_bar"],"/kaggle/input/hvdls-v1-hv-v2/checkpoint-1140",resize=True)


sub_df = pd.DataFrame(
    data={
        "id": [f"{id_}_x" for id_ in df.id.to_list()] + [f"{id_}_y" for id_ in df.id.to_list()],
        "data_series": df.pred_x.to_list() + df.pred_y.to_list(),
        "chart_type": df.pred_class.to_list() * 2,
    }
)
sub_df.data_series = sub_df.data_series.apply(lambda x: ";".join([str(y) for y in x])) 
sub_df.to_csv("submission.csv", index=False)

