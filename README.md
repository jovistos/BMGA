
# Benetech - Making Graphs Accessible Competition Solution

This repository contains the 24th place solution for the Kaggle competition called "Benetech - Making Graphs Accessible" [0] , where the goal is to extract data represented by five types of charts commonly found in STEM textbooks.

## Competition Overview

Millions of students with disabilities or learning differences face barriers in accessing educational materials, particularly when it comes to visuals like graphs. The competition, hosted by Benetech, aims to address this issue by leveraging machine learning to automatically extract data from STEM charts, making them accessible to all students.

## Solution Overview

A solution for this competition is a two-step pipeline based on Deplot model. Deplot was chosen over Matcha due to faster convergence. The pipeline consists of a simple classification task in the first step, followed by solving the task for different chart types in the second step. For every chart type a different model was used.

### Model Training

- A Deplot model was selected and fine-tuned on all chart types using competition data and synthetic data.
- For each chart type, the model was further fine-tuned using a different set of synthetic data.
- Image augmentation techniques such as RandomResize, ColorJitter, and GaussianBlur were applied to reduce overfitting.
- Random selection data points from the synthetic dataset generated by the modified script shared by @brendanartley 

### Data Preprocessing

- The dataset provided for the competition was noisy, so data cleaning was performed to stabilize training and improve accuracy.
- Several thousand samples were removed in the final version through data cleaning, although not all of them were manually selected.
- Models had a token length limit set to 720, and any samples longer than that were removed.
- Preprocessing steps included rounding numbers based on the maximum range of the corresponding axis and converting numbers from scientific notation to decimal.


## Getting Started

To reproduce or build upon this solution, follow the steps below:

1. Clone this repository: `git clone https://github.com/jovistos/BMGA.git`
2. Install the required dependencies: `conda env create -f environment.yml`
3. Download the competition dataset and place it in the `data/` directory.
4. To run the inference script and access the trained models used, see [2]


## License

This project is licensed under the [MIT License](LICENSE).

Feel free to reach out to the repository owner for any questions or further information.

- [0] https://www.kaggle.com/competitions/benetech-making-graphs-accessible/overview 
- [1] https://www.kaggle.com/datasets/brendanartley/benetech-extra-generated-data 
- [2] https://www.kaggle.com/code/joviis/deplot-infer-v24
