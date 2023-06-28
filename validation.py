import os

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from transformers import AutoProcessor, Pix2StructForConditionalGeneration

import sys
sys.path.insert(1, '/home/jovis/Documents/WORK/Kaggle/BMGA/core/data')

from data import BeneData

import numpy as np
import pandas as pd

from rapidfuzz.distance.Levenshtein import distance as levenshtein
from sklearn.metrics import r2_score

import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def sigmoid(x):
    return 2 - 2 / (1 + np.exp(-x))


def normalized_rmse(y_true, y_pred):
    # The argument to the sigmoid transform is equal to 
    # rmse(y_true, y_pred) / rmse(y_true, np.mean(y_true))
    return sigmoid((1 - r2_score(y_true, y_pred)) ** 0.5)


def normalized_levenshtein_score(y_true, y_pred):
    total_distance = np.sum([levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)


def score_series(y_true, y_pred):
    if len(y_true) != len(y_pred):
        return 0.0
    if isinstance(y_true[0], str):
        return normalized_levenshtein_score(y_true, y_pred)
    else:
        return normalized_rmse(y_true, y_pred)


def benetech_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """Evaluate predictions using the metric from the Benetech - Making Graphs Accessible.
    
    Parameters
    ----------
    ground_truth: pd.DataFrame
        Has columns `[data_series, chart_type]` and an index `id`. Values in `data_series` 
        should be either arrays of floats or arrays of strings.
    
    predictions: pd.DataFrame
    """
    if not ground_truth.index.equals(predictions.index):
        raise ValueError("Must have exactly one prediction for each ground-truth instance.")
    if not ground_truth.columns.equals(predictions.columns):
        raise ValueError(f"Predictions must have columns: {ground_truth.columns}.")
    pairs = zip(ground_truth.itertuples(index=False), predictions.itertuples(index=False))
    scores = []
    for (gt_series, gt_type), (pred_series, pred_type) in pairs:
        if gt_type != pred_type:  # Check chart_type condition
            scores.append(0.0)
        else:  # Score with RMSE or Levenshtein as appropriate
            scores.append(score_series(gt_series, pred_series))
    return np.mean(scores)


def convert_to_float(values):
    converted_values = []
    for i in values:
        try:
            value = float(i)
            if math.isnan(value):
                value=0.
        except:
            value = 0.
        converted_values.append(value)
    return converted_values

def convert_true_and_pred_to_float(df):

    df['y'] = df.apply(lambda row:
        convert_to_float(row.y) if row.chart_type == "vertical_bar" else row.y, axis=1)

    df['pred_y'] = df.apply(lambda row:
        convert_to_float(row.pred_y) if row.chart_type == "vertical_bar" else row.pred_y, axis=1)


    df['x'] = df.apply(lambda row:
        convert_to_float(row.x) if row.chart_type == "horizontal_bar" else row.x, axis=1)

    df['pred_x'] = df.apply(lambda row:
        convert_to_float(row.pred_x) if row.chart_type == "horizontal_bar" else row.pred_x, axis=1)
    

    df['y'] = df.apply(lambda row:
        convert_to_float(row.y) if row.chart_type == "line" else row.y, axis=1)

    df['pred_y'] = df.apply(lambda row:
        convert_to_float(row.pred_y) if row.chart_type == "line" else row.pred_y, axis=1)


    df['y'] = df.apply(lambda row:
        convert_to_float(row.y) if row.chart_type == "scatter" else row.y, axis=1)

    df['pred_y'] = df.apply(lambda row:
        convert_to_float(row.pred_y) if row.chart_type == "scatter" else row.pred_y, axis=1)

    df['x'] = df.apply(lambda row:
        convert_to_float(row.x) if row.chart_type == "scatter" else row.x, axis=1)

    df['pred_x'] = df.apply(lambda row:
        convert_to_float(row.pred_x) if row.chart_type == "scatter" else row.pred_x, axis=1)
    
    return df


def inference_ds(model,ds):
  
    data_loader = DataLoader(
        ds, batch_size=16, shuffle=False, num_workers=6
    )


    all_generations = []
    for batch in tqdm(data_loader):
        flattened_patches = batch["flattened_patches"].to(device).type(torch.bfloat16)
        attention_mask = batch["attention_mask"].to(device).type(torch.bfloat16)

        batch_size = flattened_patches.shape[0]
    
        
        try:
            generated_ids = model.generate(
                flattened_patches=flattened_patches, 
                attention_mask=attention_mask, 
                max_length=730,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                )


            all_generations.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
        except:
            all_generations.extend([""]*batch_size)
    return all_generations

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


def infer_df(df,model):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model.eval()
    
    # model.to(device)
    model = model.cuda().to(torch.bfloat16).to(device)
    
    # ds = conver_to_ds(df[["image_path","labels"]])
    ds = data_pipe.conver_to_ds(df[["image_path","labels"]],is_valid=True)
    
    all_generations = inference_ds(model,ds)
    df["pred_xy"]=all_generations
    
    
    # df = df.rename(columns={'chart-type': 'chart_type'})
    df.id = df.id.apply(lambda x:[x+"_x",x+"_y"])
    df["pred_x"] = df.pred_xy.apply(lambda x:xy_to_x(x,include_count=True,dim=0))
    df["pred_y"] = df.pred_xy.apply(lambda x:xy_to_x(x,include_count=True,dim=1))

    df = convert_true_and_pred_to_float(df)

    df["scores_x"]=df.apply(lambda row:score_series(row.x,row.pred_x),axis=1)
    df["scores_y"]=df.apply(lambda row:score_series(row.y,row.pred_y),axis=1)
    return df

model_name = "google/deplot"
model_path_xy= "/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/output_hvdls/hvdls_v1/checkpoint-49200"


processor = AutoProcessor.from_pretrained(model_name,is_vqa=False)
model = Pix2StructForConditionalGeneration.from_pretrained(model_path_xy)

data_pipe = BeneData(processor=processor,
                     logger=logger,
                     labels_type="xy",
                     labels_max_length=730, #512 
                     frac_gen_train=1,
                    #  classes_to_use=['line', 'horizontal_bar', 'scatter', 'vertical_bar',"dot"],
                    
                     classes_to_use=["horizontal_bar","vertical_bar"],
                    sort_axis=False,

                     use_augmentation = False,
                     use_synth=False,
                     
                
                    synth_paths= {"scatter":"/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/data/synth_data/scatter/v5",
                                "horizontal_bar":"/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/data/synth_data/horizontal_bar/v1_10000_5_40",
                                "vertical_bar":"/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/data/synth_data/vertical_bar/v1_1000_30_50",
                                "dot":"/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/data/synth_data/vertical_bar/v1_1000_30_50",
                                "line":"/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/data/synth_data/line/v3"},
                     synth_numbers = {"scatter":5000,
                                "horizontal_bar":9500,
                                "dot":0,
                                "vertical_bar":800,
                                "line":1000}
                    )

train, valid = data_pipe.get_ds_splits()

device = torch.device("cuda:1")

df = data_pipe.valid_df.copy(deep=True)

df = infer_df(df,model)

for chart_type in df.chart_type.unique().tolist():
    print(chart_type)
    
    
    score_x = df.scores_x[df.chart_type==chart_type].mean()
    score_y = df.scores_y[df.chart_type==chart_type].mean()
    print("score x",score_x)
    print("score x",score_y)
    print("*******************************")