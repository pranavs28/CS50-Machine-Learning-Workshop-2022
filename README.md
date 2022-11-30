# CS50 Machine Learning Workshop 2022
Content from the Fall 2022 CS50 Machine Learning Workshop held at Yale University.

---
## Regression

In this folder, you will find 3 files: `data.csv` contains a mock dataset of sales data corresponding to TV, Radio, and Newspaper ad spending. `single_attribute_regression.py` implements a linear regression model just accounting for TV spending. `full_regression.py` implements a regression model considering 
all attributes (TV, Radio, and Newspaper). 

---
## Classification

In this folder, you will find 2 files: `fruitdata.txt` contains a mock dataset of fruits and their names, subtype, mass, width, height, and color score. 
`classification.py` implements a K-Nearest Neighbor classifier for predicting the type of fruit given input attributes.

---
## Generation

In this folder, you will find two files: a `shakespeare.txt` that features a list Shakespeare sonnets. `generator.py` implements a GPT-2 finetuned model for generating Shakespearian text. The training step is very computationally expensive so it is recommended to instead run the code in the linked Google Colaboratory, where you can use a GPU to speed up the process.