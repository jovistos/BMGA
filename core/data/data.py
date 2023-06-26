import os
import json

from pathlib import Path
from typing import  Dict, Union

import numpy as np
import pandas as pd
from PIL import Image


from datasets import Dataset
from datasets import Image as ds_img



from transformers import AutoProcessor, Pix2StructForConditionalGeneration




from transformers import  Seq2SeqTrainer,Seq2SeqTrainingArguments, EarlyStoppingCallback



import ast

from sklearn.model_selection import train_test_split


from torchvision.transforms import Compose, ColorJitter,GaussianBlur,Resize

from torchvision.transforms.v2 import RandomResize

import random
import pickle

import swifter



DATA_DIR = "/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/data/benetech-making-graphs-accessible/train"

AUG_DATA_DIR = "/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/data/archive"

class BeneData:
    def __init__(self, processor = None,
                 labels_type = "xy",             
                 labels_max_length = 256,
                 frac_gen_train = 1,
                 include_scatter = True,
                 data_dir=Path(DATA_DIR),
                 aug_data_dir=Path(AUG_DATA_DIR),
                 classes_to_use=[],
                 use_synth=True,
                 use_augmentation=True,
                 sort_axis=True,
                 synth_paths= {"scatter":"path",
                                "horizontal_bar":"path",
                                "vertical_bar":"path",
                                "line":"path"},
                 synth_numbers = {"scatter":40000,
                                "horizontal_bar":10000,
                                "vertical_bar":10000,
                                "line":20000}):
                
        self.processor = processor
        self.labels_type = labels_type 
        self.labels_max_length = labels_max_length
        self.data_dir = data_dir
        self.aug_data_dir = aug_data_dir
        self.images_path = self.data_dir / "images"
        self.train_json_files = list((self.data_dir / "annotations").glob("*.json"))
        self.seed = 42
        self.frac_gen_train = frac_gen_train
        self.include_scatter = include_scatter
        self.classes_to_use = classes_to_use
        self.synth_numbers = synth_numbers
        self.use_synth = use_synth
        self.synth_paths = synth_paths 
        self.use_augmentation = use_augmentation
        self.sort_axis = sort_axis
        
        self.cp_lists_names = [
                                'scatter_v1_extr_cp.pickle',
                                'vert_bar_v1_val_cp.pickle',
                                'line_v1.pickle',
                                'line_v2_gen_t09.pickle',
                                "l_gen_ext_t009_cp_v3.pickle",
                                'vertical_v1_gen_cp.pickle',
                                'horz_bar_v1_cp.pickle',
                                'vert_bar_v1_extr_cp.pickle',
                                'line_v1_ext_.pickle',
                                'scatter_v1_gen_t0109.pickle',
                                'vert_bar_v1_gen_t03092_ready.pickle',
                                'vh_gen_ext_t09095_v2.pickle',
                                'vh_gen_ext_t009_cp_v2.pickle'
                                    ]

        self.SAVE_DIR_REMOVE = "/home/jovis/Documents/WORK/Kaggle/Benetech_Making_Graphs_Accessible/data/samples_to_remove"
        
                                   
    
    def load_cp(self):
        """
        Load and merge lists saved in pickle files containing sample ids to be removed.
        """
        cp_merged = []
        for file_name in self.cp_lists_names:
            path = os.path.join(self.SAVE_DIR_REMOVE,file_name)
            with open(path, "rb") as fp:   # Unpickling
                 cp_merged.extend(pickle.load(fp))
        self.samples_to_remove = cp_merged

    
    
    def are_all_floats(self, l):
        """
        Checks if all the elements in a list can be converted to floats
        """
        try:
            fl_l = [float(x) for x in l]
            return True
        except:
            return False
        
    def max_range(self, l):
        fl_l = [float(x) for x in l]
        try:
            l_range = max(fl_l)-min(fl_l)
            return [l_range]
        except:
            return []
         

    def round_sample_v2(self,l):
        if self.are_all_floats(l): 
            if bool(self.max_range(l)) & (self.max_range(l)[0]>0):
                max_range = self.max_range(l)[0]
                l = [float(x) for x in l]
                if max_range<0.0001:
                    l = [round(x,6) for x in l]
                elif max_range<0.001:
                    l = [round(x,5) for x in l]
                elif max_range<0.01:
                    l = [round(x,4) for x in l]
                elif max_range<1:
                    l = [round(x,3) for x in l]
                elif max_range<2:
                    l = [round(x,2) for x in l]
                elif max_range<30:
                    l = [round(x,1) for x in l]
                else:
                    l = [round(x,0) for x in l]
                l = [ (int(x) if x.is_integer() else x) for x in l ]
                l = [(np.format_float_positional(x) if (abs(x)<1) else x) for x in l]
                l = [str(x) for x in l]
                return l
            else:
                return l
        else:
            return l


    
    def is_nan(self,value: Union[int, float, str]) -> bool:
        """
        Check if a value is NaN (not a number).

        Args:
            value (int, float, str): The value to check

        Returns:
            bool: True if the value is NaN, False otherwise
        """
        return isinstance(value, float) and str(value) == "nan"




    def get_gt_string_and_xy(self,filepath: Union[str, os.PathLike]) -> Dict[str, str]:
        """
        Get the ground truth string and x-y data from the given JSON file.

        Args:
            filepath (str): The path to the JSON file

        Returns:
            dict: A dictionary containing the ground truth string, x-y data, chart type, id, and source
        """
        filepath = Path(filepath)

        with open(filepath) as fp:
            data = json.load(fp)

        data_series = data["data-series"]

        all_x, all_y = [], []

        for d in data_series:
            x = d["x"]
            y = d["y"]

            if self.is_nan(x) or self.is_nan(y):
                continue

            all_x.append(x)
            all_y.append(y)


        chart_type = f"<{data['chart-type']}>"
        x_str = ";".join(list(map(str, all_x))) 
        y_str = ";".join(list(map(str, all_y))) 

        gt_string =  chart_type + "&"+ x_str +"&"+ y_str

        row_dict = {
            "image_path":str(self.images_path / f"{filepath.stem}.jpg"),
            "ground_truth": gt_string,
            "x": json.dumps(all_x),
            "y": json.dumps(all_y),
            "chart_type": data["chart-type"],
            "id": filepath.stem,
            "source": data["source"],
        }

    
        return pd.DataFrame([row_dict])   

   
    def tokenize(self,examples):
        
        
        
            
        tokenized_examples = self.processor(images = examples["flattened_patches"], 
                
                           add_special_tokens=True,
                        return_tensors="pt")

        examples["flattened_patches"] = tokenized_examples["flattened_patches"]
        examples["attention_mask"] = tokenized_examples["attention_mask"]

        examples["labels"] = self.processor.tokenizer(text=examples["labels"],    
                                                     padding="max_length", 
                                                     truncation=True, 
                                                     return_tensors="pt", 
                                                     add_special_tokens=True, 
                                                     max_length=self.labels_max_length).input_ids
        return  examples


    def tokenize_train(self,examples):
        if self.use_augmentation == True:
            examples["flattened_patches"] = [self.rescale(image) for image in examples["flattened_patches"]]
            examples["flattened_patches"] = [self.transforms(image.convert("RGB")) for image in examples["flattened_patches"]]
        
        return self.tokenize(examples)
    

    def tokenize_valid(self,examples):

        examples["flattened_patches"] = [self.val_transform(image.convert("RGB")) for image in examples["flattened_patches"]]
    
        return self.tokenize(examples)
    

    def xy_to_x_augment(self,xy,dim = 0):
        try:
            # x = [y.split("^")[dim] for y in xy.split(";")]
            x = [y.split("<0x0A>")[dim].strip() for y in xy.split("|")]
            # x = [y.split("|")[dim].strip() for y in xy.split("<0x0A>")]
            # x = xy.split("&")[dim].split(";")
        except:
            x = []
        return x


    def prepare_synth_df(self, df_aug):
        
        df_aug["x"]=df_aug.text.apply(lambda x:self.xy_to_x_augment(x,dim=0))
        df_aug["y"]=df_aug.text.apply(lambda x:self.xy_to_x_augment(x,dim=1))


        df_aug.x = df_aug.x.apply(self.round_sample_v2)
        df_aug.y = df_aug.y.apply(self.round_sample_v2)

        df_aug.x = df_aug.x.apply(lambda x: [str(i) for i in x])
        df_aug.y = df_aug.y.apply(lambda x: [str(i) for i in x])

        # df_aug["image_path"]=df_aug.file_name.apply(lambda x: os.path.join(self.aug_data_dir,x))
        
        self.df_aug = df_aug


    def get_aug_df_v2(self):

        
        dfs = []
        for i in self.classes_to_use:
            df = pd.read_csv(os.path.join(self.synth_paths[i],"metadata.csv")).sample(n=self.synth_numbers[i])
            df["chart_type"]=i
            df["image_path"]=df.file_name.apply(lambda x: os.path.join(self.synth_paths[i],x))
            dfs.append(df)
        df_synth =pd.concat(dfs,axis=0).sample(frac=1,random_state=self.seed+1).reset_index(drop=True)
        self.prepare_synth_df(df_synth)


    def sort_x_y(self,x,y,chart_type):
        if self.are_all_floats(x) & (chart_type=="scatter"):
            fl_x = [float(i) for i in x]
            if self.are_all_floats(y):
                fl_y = [float(i) for i in y]
                combined = zip(fl_x,fl_y)
                combined = sorted(combined, key=lambda tup: (tup[0],tup[1]))
            else:
                combined = zip(fl_x,y)
                combined = sorted(combined, key=lambda tup: tup[0])
            return combined
        else:
            return list(zip(x,y))

    def get_df(self):

        column_dict = {"images_path":self.train_json_files}
        df  =pd.DataFrame.from_dict(column_dict)

        
        df["data"] = df.images_path.swifter.apply(self.get_gt_string_and_xy)
        

        df = pd.concat(df["data"].to_list(), ignore_index=True)   

        df = df.drop(['ground_truth'], axis=1)
        

        df.x = df.x.apply(lambda x:ast.literal_eval(x) )
        df.y = df.y.apply(lambda x:ast.literal_eval(x) )

        df.x = df.x.apply(lambda x: [str(i) for i in x])
        df.y = df.y.apply(lambda x: [str(i) for i in x])
        
        if self.sort_axis:
            df["sorted_xy"] = df.apply(lambda df_: self.sort_x_y(df_.x,df_.y,df_.chart_type),axis=1)
            df["x"] = df.sorted_xy.apply(lambda x: [i[0] for i in x])
            df["y"] = df.sorted_xy.apply(lambda x: [i[1] for i in x])

        df.x = df.x.apply(self.round_sample_v2)
        df.y = df.y.apply(self.round_sample_v2)

        df.x = df.x.apply(lambda x: [str(i) for i in x])
        df.y = df.y.apply(lambda x: [str(i) for i in x])


        self.df = df.reset_index(drop=True)


        print("df constucted")

    def set_labels(self):

        if self.labels_type == "xy":
            self.df['xy'] = self.df.apply(lambda x: list(zip(x.x,x.y)), axis=1)
            self.df["labels"] = self.df["xy"].apply(lambda x: ["^".join(i) for i in x])
            self.df["labels"] = self.df["labels"].apply(lambda x: ";".join(x))
            self.df["labels"] = self.df.apply(lambda df_:str(len(df_.x))+"**"+df_.labels,axis=1)

            # self.df['labels'] = self.df.apply(lambda df_: ";".join(df_.x)+"^"+";".join(df_.y), axis=1 )
        elif self.labels_type == "chart_type":
            self.df["labels"] = self.df["chart_type"]
        print("seting labels done!")

        # self.df["labels"] = self.df["labels_text"].swifter.apply(self.tokenize_sample)

    def token_length(self,text):
        text = self.processor.tokenizer(text=text,     
                                    truncation=False, 
                                        return_tensors="pt", 
                                            add_special_tokens=True, 
                                                     ).input_ids[0]

        return len(text)

    def get_df_splits(self):

        self.get_df()
        
        self.get_aug_df_v2()
        if self.use_synth==True:
            
            self.df = pd.concat([self.df,self.df_aug],axis=0).reset_index(drop=True)
            

        
        self.set_labels()

        self.df["token_length"] = self.df.labels.swifter.apply(self.token_length)

        numb_removed = len(self.df[self.df.token_length>=self.labels_max_length])

        self.df= self.df[self.df.token_length<self.labels_max_length].reset_index(drop=True) 

        
        print("length to high. Numb of tokens removed",numb_removed)


        change = self.df[(self.df.source=="generated")&(self.df.chart_type=="dot")].sample(frac=0.1,random_state=self.seed).index
        self.df.loc[change,'source'] = "extracted"

        df_g = self.df[self.df.source!="extracted"]
        self.df_e = self.df[self.df.source=="extracted"].reset_index(drop=True)

        train_e, valid_df = train_test_split(self.df_e, test_size=0.1, random_state=self.seed)

        self.df_g = df_g.sample(frac=self.frac_gen_train,random_state=self.seed).reset_index(drop=True)

        train_df = pd.concat([self.df_g, train_e], axis=0).sample(frac=1,random_state=self.seed).reset_index(drop=True)
        
        
        train_df = train_df[train_df.chart_type.isin(self.classes_to_use)]    
        valid_df = valid_df[valid_df.chart_type.isin(self.classes_to_use)]  


        self.load_cp()
        for sample_id in self.samples_to_remove:
            
            train_df = train_df[train_df.id != sample_id] 
            valid_df = valid_df[valid_df.id != sample_id] 

        self.train_df = train_df.reset_index(drop=True)
        self.valid_df = valid_df.reset_index(drop=True)
        
    
    def conver_to_ds(self,dff,is_valid=False):

        ds = Dataset.from_pandas(dff.copy().reset_index(drop=True))
        ds = ds.cast_column("image_path", ds_img())
        ds = ds.rename_column("image_path", "flattened_patches")
        if is_valid:
            ds.set_transform(self.tokenize_valid) 
        else:
            ds.set_transform(self.tokenize_train)

       

        return ds

    def rescale(self,img,strech_factor=0.3):
        width, height = img.size
        rnd1 = random.uniform(-strech_factor, strech_factor)
        new_w = width + int(width*rnd1)
        rnd2 = random.uniform(-strech_factor, strech_factor)
        new_h = height + int(height*rnd2)

        transform = Resize((new_h,new_w))

        img = transform(img)
    
        return(img)
    
    def get_ds_splits(self):

        self.transforms = Compose([ 
                    RandomResize(min_size=400, max_size=600, antialias=True),
                    ColorJitter(brightness=0.1, hue=0.5,contrast=0.1,saturation=0.8),
                    GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 5)),
                            ])
        self.val_transform = Resize(size=600, antialias=True)
                    

        self.get_df_splits()
        self.train_ds = self.conver_to_ds(self.train_df[["image_path","labels"]])
        self.valid_ds = self.conver_to_ds(self.valid_df[["image_path","labels"]],is_valid=True)

        return self.train_ds,self.valid_ds