
import os, sys, glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import List



# CUDA for PyTorch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print("Using Device:  ", device)


class MyCustomDataset(torch.utils.data.Dataset):
    
    def __init__(self, im_dir_path = "/data/Regular/images/",  annotation_dir_path="/data/Regular/annotations/",
                im_mode='L', transform = None):
        
        self.im_dir_path = im_dir_path
        self.annotation_dir_path = annotation_dir_path
        self.im_mode = im_mode
        self.transform = transform
        
        self.data_frames = []
        self.load_data_frames() 
        
    
    def load_data_frames(self):
        im_paths, img_files_id = self.collect_file_paths(self.im_dir_path, ".png")
        text_paths, text_files_id = self.collect_file_paths(self.annotation_dir_path, ".txt")
        
        for i, im_path in enumerate(im_paths):
            try:
                text_path = text_paths[text_files_id.index(img_files_id[i])]
                self.data_frames += [{"im_path":im_path, "text_path":text_path}]
            except ValueError:
                print("No Annotation File Found Against Image File : ",  im_path)
        
        print("Created Dataframe With Length : ", len(self.data_frames))
        
    def collect_file_paths(self, dir_path: str, file_extensions: str, sort: bool = True) ->  (List[str], List[str]):
        cwd = os.getcwd()
        full_path = cwd + dir_path + "*" + file_extensions
        files_path = glob.glob(full_path,recursive=True)
        files_id = [file_path.split("/")[-1].split(".")[0] for file_path in files_path]

        return files_path, files_id

    
    def __len__(self):
        return len(self.data_frames)
    
    def __getitem__(self, index):
        
        df = self.data_frames[index]
        
        im_path = df["im_path"]
        txt_path = df["text_path"]
        
        x = Image.open(im_path).convert(self.im_mode)
        with open(txt_path, 'r') as file:
            y = file.read()
         
        if self.transform is None:
            self.transform = transforms.ToTensor()
        x_trfm = self.transform(x)

        return {"x": x_trfm, "y": y}


dataloader_hyperparam  = {
    'batch_size': 4,
    'shuffle' : True
}

my_custom_dataset = MyCustomDataset()
transforms.ToPILImage()(my_custom_dataset[74]['x']).show()
print("Text of Image : ", my_custom_dataset[74]['y'])
# data_generator = torch.utils.data.DataLoader(dataloader, **DataLoader_params)

