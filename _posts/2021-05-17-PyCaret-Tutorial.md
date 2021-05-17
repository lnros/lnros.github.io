---
title: "PyCaret Tutorial"
date: 2021-05-17
categories:
  - machine-learning
tags:
  - pycaret
  - automl
---

# PyCaret Tutorial

This tutorial will not cover PyCaret as a whole, but it will rather focus on the entire pipeline for a single classification problem. We will use the wine recognition dataset:

Lichman, M. (2013). [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml). Irvine, CA: University of California, School of Information and Computer Science.


> OBS: This post was automatically generated from my [Jupyter Notebook](https://github.com/lnros/lnros.github.io/blob/master/notebooks/PyCaret_Tutorial.ipynb) thanks to Adam Blomberg's [post](https://blomadam.github.io/tutorials/2017/04/09/ipynb-to-Jekyll-Post-tools.html).

## What is PyCaret?

PyCaret is a library that comes to automate the Machine Learning (ML) process requiring very few lines of code. It works with multiple other useful ML libraries, such as scikit-learn.

Check out their website for more information: [PyCaret Homepage](https://pycaret.org).

## Installing PyCaret

### Virtual environment

First, we create a virtual environment to ensure we do not have unnecessary packages as well as to keep things isolated and light.

Using `venv`:

```shell
$ python -m venv /path/to/new/virtual/environment
```

And to activate it:

```shell
$ source /path/to/new/virtual/environment/bin activate
```

Using `conda`:

```shell
$ conda create --name venv pip
```

And to activate it:

```shell
$ conda source venv activate
```

### Actual PyCaret installation

We use `pip` to install it:

```shell
$ pip install pycaret
```
Or on the notebook:

```notebook
! pip install pycaret
```

## Dataset

We will import the dataset from Scikit-learn:


```python
import sklearn
from sklearn.datasets import load_wine
print(f"Scikit-learn version: {sklearn.__version__}")
import pandas as pd
import numpy as np
```

    Scikit-learn version: 0.23.2



```python
data = load_wine()
df = pd.DataFrame(data=np.c_[data['data'], data['target']],
                  columns=data['feature_names'] + ['target'])
df.shape
```




    (178, 14)



Let's do some _**very brief**_ EDA on the dataset:


```python
display(df.head())
df.describe()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113.0</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118.0</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.000618</td>
      <td>2.336348</td>
      <td>2.366517</td>
      <td>19.494944</td>
      <td>99.741573</td>
      <td>2.295112</td>
      <td>2.029270</td>
      <td>0.361854</td>
      <td>1.590899</td>
      <td>5.058090</td>
      <td>0.957449</td>
      <td>2.611685</td>
      <td>746.893258</td>
      <td>0.938202</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.811827</td>
      <td>1.117146</td>
      <td>0.274344</td>
      <td>3.339564</td>
      <td>14.282484</td>
      <td>0.625851</td>
      <td>0.998859</td>
      <td>0.124453</td>
      <td>0.572359</td>
      <td>2.318286</td>
      <td>0.228572</td>
      <td>0.709990</td>
      <td>314.907474</td>
      <td>0.775035</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.030000</td>
      <td>0.740000</td>
      <td>1.360000</td>
      <td>10.600000</td>
      <td>70.000000</td>
      <td>0.980000</td>
      <td>0.340000</td>
      <td>0.130000</td>
      <td>0.410000</td>
      <td>1.280000</td>
      <td>0.480000</td>
      <td>1.270000</td>
      <td>278.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.362500</td>
      <td>1.602500</td>
      <td>2.210000</td>
      <td>17.200000</td>
      <td>88.000000</td>
      <td>1.742500</td>
      <td>1.205000</td>
      <td>0.270000</td>
      <td>1.250000</td>
      <td>3.220000</td>
      <td>0.782500</td>
      <td>1.937500</td>
      <td>500.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.050000</td>
      <td>1.865000</td>
      <td>2.360000</td>
      <td>19.500000</td>
      <td>98.000000</td>
      <td>2.355000</td>
      <td>2.135000</td>
      <td>0.340000</td>
      <td>1.555000</td>
      <td>4.690000</td>
      <td>0.965000</td>
      <td>2.780000</td>
      <td>673.500000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.677500</td>
      <td>3.082500</td>
      <td>2.557500</td>
      <td>21.500000</td>
      <td>107.000000</td>
      <td>2.800000</td>
      <td>2.875000</td>
      <td>0.437500</td>
      <td>1.950000</td>
      <td>6.200000</td>
      <td>1.120000</td>
      <td>3.170000</td>
      <td>985.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.830000</td>
      <td>5.800000</td>
      <td>3.230000</td>
      <td>30.000000</td>
      <td>162.000000</td>
      <td>3.880000</td>
      <td>5.080000</td>
      <td>0.660000</td>
      <td>3.580000</td>
      <td>13.000000</td>
      <td>1.710000</td>
      <td>4.000000</td>
      <td>1680.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>



All data is numeric and without missing values.


```python
# Checking the classes
df['target'].value_counts()
```




    1.0    71
    0.0    59
    2.0    48
    Name: target, dtype: int64



The classes are not very imbalanced.

**Splitting the dataset into train and test**


```python
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df,
                                     test_size=0.20,
                                     random_state=0,
                                     stratify=df['target'])
```

## Initializing PyCaret

We can, now, import PyCaret and start playing around with it!


```python
import pycaret
from pycaret.classification import *
print(f"PyCaret version: {pycaret.__version__}")
```

    PyCaret version: 2.3.1


PyCaret syntax asks us to setup the data as the first step, providing the input data with the features and the target variable. This `setup` function is how PyCaret initializes the pipeline for future preprocessing, modeling, and deployment. It requires two parameters: `data`, a pandas dataframe, and `target`, the name of the target column. There are other parameters, but they are all optional and we will not cover them at this point.

The `setup` infers on its own the features' data types, but it is worth double checking since it may do it incorrectly sometimes. A table showing the features and their data types is displayed after running the `setup`. It asks us to confirm if correct and press enter. If something is incorrect, then we can correct it by typing `quit`. Otherwise, we can proceed. Notice that getting the data types right is extremely important, because PyCaret automatically preprocesses the data based on each feature's type. Preprocessing is a fundamental and critical part of doing ML properly.


```python
# setup the dataset
grid = setup(data=df_train, target='target')
```


<style  type="text/css" >
</style><table id="T_7b34b_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Description</th>        <th class="col_heading level0 col1" >Value</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_7b34b_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_7b34b_row0_col0" class="data row0 col0" >session_id</td>
                        <td id="T_7b34b_row0_col1" class="data row0 col1" >6420</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_7b34b_row1_col0" class="data row1 col0" >Target</td>
                        <td id="T_7b34b_row1_col1" class="data row1 col1" >target</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_7b34b_row2_col0" class="data row2 col0" >Target Type</td>
                        <td id="T_7b34b_row2_col1" class="data row2 col1" >Multiclass</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_7b34b_row3_col0" class="data row3 col0" >Label Encoded</td>
                        <td id="T_7b34b_row3_col1" class="data row3 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_7b34b_row4_col0" class="data row4 col0" >Original Data</td>
                        <td id="T_7b34b_row4_col1" class="data row4 col1" >(142, 14)</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_7b34b_row5_col0" class="data row5 col0" >Missing Values</td>
                        <td id="T_7b34b_row5_col1" class="data row5 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_7b34b_row6_col0" class="data row6 col0" >Numeric Features</td>
                        <td id="T_7b34b_row6_col1" class="data row6 col1" >13</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_7b34b_row7_col0" class="data row7 col0" >Categorical Features</td>
                        <td id="T_7b34b_row7_col1" class="data row7 col1" >0</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_7b34b_row8_col0" class="data row8 col0" >Ordinal Features</td>
                        <td id="T_7b34b_row8_col1" class="data row8 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_7b34b_row9_col0" class="data row9 col0" >High Cardinality Features</td>
                        <td id="T_7b34b_row9_col1" class="data row9 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_7b34b_row10_col0" class="data row10 col0" >High Cardinality Method</td>
                        <td id="T_7b34b_row10_col1" class="data row10 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_7b34b_row11_col0" class="data row11 col0" >Transformed Train Set</td>
                        <td id="T_7b34b_row11_col1" class="data row11 col1" >(99, 13)</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_7b34b_row12_col0" class="data row12 col0" >Transformed Test Set</td>
                        <td id="T_7b34b_row12_col1" class="data row12 col1" >(43, 13)</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_7b34b_row13_col0" class="data row13 col0" >Shuffle Train-Test</td>
                        <td id="T_7b34b_row13_col1" class="data row13 col1" >True</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_7b34b_row14_col0" class="data row14 col0" >Stratify Train-Test</td>
                        <td id="T_7b34b_row14_col1" class="data row14 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_7b34b_row15_col0" class="data row15 col0" >Fold Generator</td>
                        <td id="T_7b34b_row15_col1" class="data row15 col1" >StratifiedKFold</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_7b34b_row16_col0" class="data row16 col0" >Fold Number</td>
                        <td id="T_7b34b_row16_col1" class="data row16 col1" >10</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_7b34b_row17_col0" class="data row17 col0" >CPU Jobs</td>
                        <td id="T_7b34b_row17_col1" class="data row17 col1" >-1</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_7b34b_row18_col0" class="data row18 col0" >Use GPU</td>
                        <td id="T_7b34b_row18_col1" class="data row18 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_7b34b_row19_col0" class="data row19 col0" >Log Experiment</td>
                        <td id="T_7b34b_row19_col1" class="data row19 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_7b34b_row20_col0" class="data row20 col0" >Experiment Name</td>
                        <td id="T_7b34b_row20_col1" class="data row20 col1" >clf-default-name</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_7b34b_row21_col0" class="data row21 col0" >USI</td>
                        <td id="T_7b34b_row21_col1" class="data row21 col1" >6d1b</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_7b34b_row22_col0" class="data row22 col0" >Imputation Type</td>
                        <td id="T_7b34b_row22_col1" class="data row22 col1" >simple</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_7b34b_row23_col0" class="data row23 col0" >Iterative Imputation Iteration</td>
                        <td id="T_7b34b_row23_col1" class="data row23 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_7b34b_row24_col0" class="data row24 col0" >Numeric Imputer</td>
                        <td id="T_7b34b_row24_col1" class="data row24 col1" >mean</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_7b34b_row25_col0" class="data row25 col0" >Iterative Imputation Numeric Model</td>
                        <td id="T_7b34b_row25_col1" class="data row25 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_7b34b_row26_col0" class="data row26 col0" >Categorical Imputer</td>
                        <td id="T_7b34b_row26_col1" class="data row26 col1" >constant</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_7b34b_row27_col0" class="data row27 col0" >Iterative Imputation Categorical Model</td>
                        <td id="T_7b34b_row27_col1" class="data row27 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row28" class="row_heading level0 row28" >28</th>
                        <td id="T_7b34b_row28_col0" class="data row28 col0" >Unknown Categoricals Handling</td>
                        <td id="T_7b34b_row28_col1" class="data row28 col1" >least_frequent</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row29" class="row_heading level0 row29" >29</th>
                        <td id="T_7b34b_row29_col0" class="data row29 col0" >Normalize</td>
                        <td id="T_7b34b_row29_col1" class="data row29 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row30" class="row_heading level0 row30" >30</th>
                        <td id="T_7b34b_row30_col0" class="data row30 col0" >Normalize Method</td>
                        <td id="T_7b34b_row30_col1" class="data row30 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_7b34b_row31_col0" class="data row31 col0" >Transformation</td>
                        <td id="T_7b34b_row31_col1" class="data row31 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row32" class="row_heading level0 row32" >32</th>
                        <td id="T_7b34b_row32_col0" class="data row32 col0" >Transformation Method</td>
                        <td id="T_7b34b_row32_col1" class="data row32 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row33" class="row_heading level0 row33" >33</th>
                        <td id="T_7b34b_row33_col0" class="data row33 col0" >PCA</td>
                        <td id="T_7b34b_row33_col1" class="data row33 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row34" class="row_heading level0 row34" >34</th>
                        <td id="T_7b34b_row34_col0" class="data row34 col0" >PCA Method</td>
                        <td id="T_7b34b_row34_col1" class="data row34 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row35" class="row_heading level0 row35" >35</th>
                        <td id="T_7b34b_row35_col0" class="data row35 col0" >PCA Components</td>
                        <td id="T_7b34b_row35_col1" class="data row35 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row36" class="row_heading level0 row36" >36</th>
                        <td id="T_7b34b_row36_col0" class="data row36 col0" >Ignore Low Variance</td>
                        <td id="T_7b34b_row36_col1" class="data row36 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row37" class="row_heading level0 row37" >37</th>
                        <td id="T_7b34b_row37_col0" class="data row37 col0" >Combine Rare Levels</td>
                        <td id="T_7b34b_row37_col1" class="data row37 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row38" class="row_heading level0 row38" >38</th>
                        <td id="T_7b34b_row38_col0" class="data row38 col0" >Rare Level Threshold</td>
                        <td id="T_7b34b_row38_col1" class="data row38 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row39" class="row_heading level0 row39" >39</th>
                        <td id="T_7b34b_row39_col0" class="data row39 col0" >Numeric Binning</td>
                        <td id="T_7b34b_row39_col1" class="data row39 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row40" class="row_heading level0 row40" >40</th>
                        <td id="T_7b34b_row40_col0" class="data row40 col0" >Remove Outliers</td>
                        <td id="T_7b34b_row40_col1" class="data row40 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row41" class="row_heading level0 row41" >41</th>
                        <td id="T_7b34b_row41_col0" class="data row41 col0" >Outliers Threshold</td>
                        <td id="T_7b34b_row41_col1" class="data row41 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row42" class="row_heading level0 row42" >42</th>
                        <td id="T_7b34b_row42_col0" class="data row42 col0" >Remove Multicollinearity</td>
                        <td id="T_7b34b_row42_col1" class="data row42 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row43" class="row_heading level0 row43" >43</th>
                        <td id="T_7b34b_row43_col0" class="data row43 col0" >Multicollinearity Threshold</td>
                        <td id="T_7b34b_row43_col1" class="data row43 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row44" class="row_heading level0 row44" >44</th>
                        <td id="T_7b34b_row44_col0" class="data row44 col0" >Clustering</td>
                        <td id="T_7b34b_row44_col1" class="data row44 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row45" class="row_heading level0 row45" >45</th>
                        <td id="T_7b34b_row45_col0" class="data row45 col0" >Clustering Iteration</td>
                        <td id="T_7b34b_row45_col1" class="data row45 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row46" class="row_heading level0 row46" >46</th>
                        <td id="T_7b34b_row46_col0" class="data row46 col0" >Polynomial Features</td>
                        <td id="T_7b34b_row46_col1" class="data row46 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row47" class="row_heading level0 row47" >47</th>
                        <td id="T_7b34b_row47_col0" class="data row47 col0" >Polynomial Degree</td>
                        <td id="T_7b34b_row47_col1" class="data row47 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row48" class="row_heading level0 row48" >48</th>
                        <td id="T_7b34b_row48_col0" class="data row48 col0" >Trignometry Features</td>
                        <td id="T_7b34b_row48_col1" class="data row48 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row49" class="row_heading level0 row49" >49</th>
                        <td id="T_7b34b_row49_col0" class="data row49 col0" >Polynomial Threshold</td>
                        <td id="T_7b34b_row49_col1" class="data row49 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row50" class="row_heading level0 row50" >50</th>
                        <td id="T_7b34b_row50_col0" class="data row50 col0" >Group Features</td>
                        <td id="T_7b34b_row50_col1" class="data row50 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row51" class="row_heading level0 row51" >51</th>
                        <td id="T_7b34b_row51_col0" class="data row51 col0" >Feature Selection</td>
                        <td id="T_7b34b_row51_col1" class="data row51 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row52" class="row_heading level0 row52" >52</th>
                        <td id="T_7b34b_row52_col0" class="data row52 col0" >Feature Selection Method</td>
                        <td id="T_7b34b_row52_col1" class="data row52 col1" >classic</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row53" class="row_heading level0 row53" >53</th>
                        <td id="T_7b34b_row53_col0" class="data row53 col0" >Features Selection Threshold</td>
                        <td id="T_7b34b_row53_col1" class="data row53 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row54" class="row_heading level0 row54" >54</th>
                        <td id="T_7b34b_row54_col0" class="data row54 col0" >Feature Interaction</td>
                        <td id="T_7b34b_row54_col1" class="data row54 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row55" class="row_heading level0 row55" >55</th>
                        <td id="T_7b34b_row55_col0" class="data row55 col0" >Feature Ratio</td>
                        <td id="T_7b34b_row55_col1" class="data row55 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row56" class="row_heading level0 row56" >56</th>
                        <td id="T_7b34b_row56_col0" class="data row56 col0" >Interaction Threshold</td>
                        <td id="T_7b34b_row56_col1" class="data row56 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row57" class="row_heading level0 row57" >57</th>
                        <td id="T_7b34b_row57_col0" class="data row57 col0" >Fix Imbalance</td>
                        <td id="T_7b34b_row57_col1" class="data row57 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7b34b_level0_row58" class="row_heading level0 row58" >58</th>
                        <td id="T_7b34b_row58_col0" class="data row58 col0" >Fix Imbalance Method</td>
                        <td id="T_7b34b_row58_col1" class="data row58 col1" >SMOTE</td>
            </tr>
    </tbody></table>


Among the optional parameters we skipped, there were options for preprocessing such as outline removal, feature selection, feature encoding, dimensionality reduction, how to split between train and test set, and many more! Check out the documentation for more details.


## Comparing different models

After setting up, the time to compare and evaluate different models has arrived! And this is done so easily that pretty much anyone with a minimal knowledge on metrics could pick up the best model. This is one of the cool PyCaret's features that allows us to save a lot of time.


PyCaret uses 10-fold cross-validation as its default, sort results by classification accuracy, returns the best model, and displays the results of all tested classifiers with different metrics.


```python
best = compare_models()
```


<style  type="text/css" >
    #T_511ef_ th {
          text-align: left;
    }#T_511ef_row0_col0,#T_511ef_row0_col2,#T_511ef_row1_col0,#T_511ef_row1_col1,#T_511ef_row1_col2,#T_511ef_row1_col3,#T_511ef_row1_col4,#T_511ef_row1_col5,#T_511ef_row1_col6,#T_511ef_row1_col7,#T_511ef_row2_col0,#T_511ef_row2_col1,#T_511ef_row2_col2,#T_511ef_row2_col3,#T_511ef_row2_col4,#T_511ef_row2_col5,#T_511ef_row2_col6,#T_511ef_row2_col7,#T_511ef_row3_col0,#T_511ef_row3_col1,#T_511ef_row3_col2,#T_511ef_row3_col3,#T_511ef_row3_col4,#T_511ef_row3_col5,#T_511ef_row3_col6,#T_511ef_row3_col7,#T_511ef_row4_col0,#T_511ef_row4_col1,#T_511ef_row4_col3,#T_511ef_row4_col4,#T_511ef_row4_col5,#T_511ef_row4_col6,#T_511ef_row4_col7,#T_511ef_row5_col0,#T_511ef_row5_col1,#T_511ef_row5_col2,#T_511ef_row5_col3,#T_511ef_row5_col4,#T_511ef_row5_col5,#T_511ef_row5_col6,#T_511ef_row5_col7,#T_511ef_row6_col0,#T_511ef_row6_col1,#T_511ef_row6_col2,#T_511ef_row6_col3,#T_511ef_row6_col4,#T_511ef_row6_col5,#T_511ef_row6_col6,#T_511ef_row6_col7,#T_511ef_row7_col0,#T_511ef_row7_col1,#T_511ef_row7_col2,#T_511ef_row7_col3,#T_511ef_row7_col4,#T_511ef_row7_col5,#T_511ef_row7_col6,#T_511ef_row7_col7,#T_511ef_row8_col0,#T_511ef_row8_col1,#T_511ef_row8_col2,#T_511ef_row8_col3,#T_511ef_row8_col4,#T_511ef_row8_col5,#T_511ef_row8_col6,#T_511ef_row8_col7,#T_511ef_row9_col0,#T_511ef_row9_col1,#T_511ef_row9_col2,#T_511ef_row9_col3,#T_511ef_row9_col4,#T_511ef_row9_col5,#T_511ef_row9_col6,#T_511ef_row9_col7,#T_511ef_row10_col0,#T_511ef_row10_col1,#T_511ef_row10_col2,#T_511ef_row10_col3,#T_511ef_row10_col4,#T_511ef_row10_col5,#T_511ef_row10_col6,#T_511ef_row10_col7,#T_511ef_row11_col0,#T_511ef_row11_col1,#T_511ef_row11_col2,#T_511ef_row11_col3,#T_511ef_row11_col4,#T_511ef_row11_col5,#T_511ef_row11_col6,#T_511ef_row11_col7,#T_511ef_row12_col0,#T_511ef_row12_col1,#T_511ef_row12_col2,#T_511ef_row12_col3,#T_511ef_row12_col4,#T_511ef_row12_col5,#T_511ef_row12_col6,#T_511ef_row12_col7{
            text-align:  left;
            text-align:  left;
        }#T_511ef_row0_col1,#T_511ef_row0_col3,#T_511ef_row0_col4,#T_511ef_row0_col5,#T_511ef_row0_col6,#T_511ef_row0_col7,#T_511ef_row4_col2{
            text-align:  left;
            text-align:  left;
            background-color:  yellow;
        }#T_511ef_row0_col8,#T_511ef_row1_col8,#T_511ef_row2_col8,#T_511ef_row4_col8,#T_511ef_row5_col8,#T_511ef_row6_col8,#T_511ef_row7_col8,#T_511ef_row8_col8,#T_511ef_row9_col8,#T_511ef_row10_col8,#T_511ef_row11_col8,#T_511ef_row12_col8{
            text-align:  left;
            text-align:  left;
            background-color:  lightgrey;
        }#T_511ef_row3_col8{
            text-align:  left;
            text-align:  left;
            background-color:  yellow;
            background-color:  lightgrey;
        }</style><table id="T_511ef_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >Accuracy</th>        <th class="col_heading level0 col2" >AUC</th>        <th class="col_heading level0 col3" >Recall</th>        <th class="col_heading level0 col4" >Prec.</th>        <th class="col_heading level0 col5" >F1</th>        <th class="col_heading level0 col6" >Kappa</th>        <th class="col_heading level0 col7" >MCC</th>        <th class="col_heading level0 col8" >TT (Sec)</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_511ef_level0_row0" class="row_heading level0 row0" >rf</th>
                        <td id="T_511ef_row0_col0" class="data row0 col0" >Random Forest Classifier</td>
                        <td id="T_511ef_row0_col1" class="data row0 col1" >0.9700</td>
                        <td id="T_511ef_row0_col2" class="data row0 col2" >0.9975</td>
                        <td id="T_511ef_row0_col3" class="data row0 col3" >0.9750</td>
                        <td id="T_511ef_row0_col4" class="data row0 col4" >0.9770</td>
                        <td id="T_511ef_row0_col5" class="data row0 col5" >0.9698</td>
                        <td id="T_511ef_row0_col6" class="data row0 col6" >0.9545</td>
                        <td id="T_511ef_row0_col7" class="data row0 col7" >0.9585</td>
                        <td id="T_511ef_row0_col8" class="data row0 col8" >0.0790</td>
            </tr>
            <tr>
                        <th id="T_511ef_level0_row1" class="row_heading level0 row1" >nb</th>
                        <td id="T_511ef_row1_col0" class="data row1 col0" >Naive Bayes</td>
                        <td id="T_511ef_row1_col1" class="data row1 col1" >0.9600</td>
                        <td id="T_511ef_row1_col2" class="data row1 col2" >0.9950</td>
                        <td id="T_511ef_row1_col3" class="data row1 col3" >0.9639</td>
                        <td id="T_511ef_row1_col4" class="data row1 col4" >0.9678</td>
                        <td id="T_511ef_row1_col5" class="data row1 col5" >0.9598</td>
                        <td id="T_511ef_row1_col6" class="data row1 col6" >0.9394</td>
                        <td id="T_511ef_row1_col7" class="data row1 col7" >0.9431</td>
                        <td id="T_511ef_row1_col8" class="data row1 col8" >0.0060</td>
            </tr>
            <tr>
                        <th id="T_511ef_level0_row2" class="row_heading level0 row2" >et</th>
                        <td id="T_511ef_row2_col0" class="data row2 col0" >Extra Trees Classifier</td>
                        <td id="T_511ef_row2_col1" class="data row2 col1" >0.9600</td>
                        <td id="T_511ef_row2_col2" class="data row2 col2" >0.9967</td>
                        <td id="T_511ef_row2_col3" class="data row2 col3" >0.9700</td>
                        <td id="T_511ef_row2_col4" class="data row2 col4" >0.9703</td>
                        <td id="T_511ef_row2_col5" class="data row2 col5" >0.9604</td>
                        <td id="T_511ef_row2_col6" class="data row2 col6" >0.9384</td>
                        <td id="T_511ef_row2_col7" class="data row2 col7" >0.9438</td>
                        <td id="T_511ef_row2_col8" class="data row2 col8" >0.0670</td>
            </tr>
            <tr>
                        <th id="T_511ef_level0_row3" class="row_heading level0 row3" >ridge</th>
                        <td id="T_511ef_row3_col0" class="data row3 col0" >Ridge Classifier</td>
                        <td id="T_511ef_row3_col1" class="data row3 col1" >0.9589</td>
                        <td id="T_511ef_row3_col2" class="data row3 col2" >0.0000</td>
                        <td id="T_511ef_row3_col3" class="data row3 col3" >0.9600</td>
                        <td id="T_511ef_row3_col4" class="data row3 col4" >0.9686</td>
                        <td id="T_511ef_row3_col5" class="data row3 col5" >0.9578</td>
                        <td id="T_511ef_row3_col6" class="data row3 col6" >0.9365</td>
                        <td id="T_511ef_row3_col7" class="data row3 col7" >0.9423</td>
                        <td id="T_511ef_row3_col8" class="data row3 col8" >0.0050</td>
            </tr>
            <tr>
                        <th id="T_511ef_level0_row4" class="row_heading level0 row4" >lightgbm</th>
                        <td id="T_511ef_row4_col0" class="data row4 col0" >Light Gradient Boosting Machine</td>
                        <td id="T_511ef_row4_col1" class="data row4 col1" >0.9500</td>
                        <td id="T_511ef_row4_col2" class="data row4 col2" >1.0000</td>
                        <td id="T_511ef_row4_col3" class="data row4 col3" >0.9589</td>
                        <td id="T_511ef_row4_col4" class="data row4 col4" >0.9623</td>
                        <td id="T_511ef_row4_col5" class="data row4 col5" >0.9486</td>
                        <td id="T_511ef_row4_col6" class="data row4 col6" >0.9242</td>
                        <td id="T_511ef_row4_col7" class="data row4 col7" >0.9315</td>
                        <td id="T_511ef_row4_col8" class="data row4 col8" >0.1100</td>
            </tr>
            <tr>
                        <th id="T_511ef_level0_row5" class="row_heading level0 row5" >lda</th>
                        <td id="T_511ef_row5_col0" class="data row5 col0" >Linear Discriminant Analysis</td>
                        <td id="T_511ef_row5_col1" class="data row5 col1" >0.9489</td>
                        <td id="T_511ef_row5_col2" class="data row5 col2" >0.9936</td>
                        <td id="T_511ef_row5_col3" class="data row5 col3" >0.9533</td>
                        <td id="T_511ef_row5_col4" class="data row5 col4" >0.9619</td>
                        <td id="T_511ef_row5_col5" class="data row5 col5" >0.9483</td>
                        <td id="T_511ef_row5_col6" class="data row5 col6" >0.9211</td>
                        <td id="T_511ef_row5_col7" class="data row5 col7" >0.9283</td>
                        <td id="T_511ef_row5_col8" class="data row5 col8" >0.0060</td>
            </tr>
            <tr>
                        <th id="T_511ef_level0_row6" class="row_heading level0 row6" >qda</th>
                        <td id="T_511ef_row6_col0" class="data row6 col0" >Quadratic Discriminant Analysis</td>
                        <td id="T_511ef_row6_col1" class="data row6 col1" >0.9389</td>
                        <td id="T_511ef_row6_col2" class="data row6 col2" >0.9950</td>
                        <td id="T_511ef_row6_col3" class="data row6 col3" >0.9167</td>
                        <td id="T_511ef_row6_col4" class="data row6 col4" >0.9534</td>
                        <td id="T_511ef_row6_col5" class="data row6 col5" >0.9319</td>
                        <td id="T_511ef_row6_col6" class="data row6 col6" >0.9023</td>
                        <td id="T_511ef_row6_col7" class="data row6 col7" >0.9135</td>
                        <td id="T_511ef_row6_col8" class="data row6 col8" >0.0070</td>
            </tr>
            <tr>
                        <th id="T_511ef_level0_row7" class="row_heading level0 row7" >dt</th>
                        <td id="T_511ef_row7_col0" class="data row7 col0" >Decision Tree Classifier</td>
                        <td id="T_511ef_row7_col1" class="data row7 col1" >0.9200</td>
                        <td id="T_511ef_row7_col2" class="data row7 col2" >0.9369</td>
                        <td id="T_511ef_row7_col3" class="data row7 col3" >0.9167</td>
                        <td id="T_511ef_row7_col4" class="data row7 col4" >0.9330</td>
                        <td id="T_511ef_row7_col5" class="data row7 col5" >0.9178</td>
                        <td id="T_511ef_row7_col6" class="data row7 col6" >0.8773</td>
                        <td id="T_511ef_row7_col7" class="data row7 col7" >0.8855</td>
                        <td id="T_511ef_row7_col8" class="data row7 col8" >0.0060</td>
            </tr>
            <tr>
                        <th id="T_511ef_level0_row8" class="row_heading level0 row8" >lr</th>
                        <td id="T_511ef_row8_col0" class="data row8 col0" >Logistic Regression</td>
                        <td id="T_511ef_row8_col1" class="data row8 col1" >0.9189</td>
                        <td id="T_511ef_row8_col2" class="data row8 col2" >0.9888</td>
                        <td id="T_511ef_row8_col3" class="data row8 col3" >0.9194</td>
                        <td id="T_511ef_row8_col4" class="data row8 col4" >0.9314</td>
                        <td id="T_511ef_row8_col5" class="data row8 col5" >0.9175</td>
                        <td id="T_511ef_row8_col6" class="data row8 col6" >0.8762</td>
                        <td id="T_511ef_row8_col7" class="data row8 col7" >0.8830</td>
                        <td id="T_511ef_row8_col8" class="data row8 col8" >0.4420</td>
            </tr>
            <tr>
                        <th id="T_511ef_level0_row9" class="row_heading level0 row9" >gbc</th>
                        <td id="T_511ef_row9_col0" class="data row9 col0" >Gradient Boosting Classifier</td>
                        <td id="T_511ef_row9_col1" class="data row9 col1" >0.8900</td>
                        <td id="T_511ef_row9_col2" class="data row9 col2" >0.9876</td>
                        <td id="T_511ef_row9_col3" class="data row9 col3" >0.8922</td>
                        <td id="T_511ef_row9_col4" class="data row9 col4" >0.9127</td>
                        <td id="T_511ef_row9_col5" class="data row9 col5" >0.8883</td>
                        <td id="T_511ef_row9_col6" class="data row9 col6" >0.8290</td>
                        <td id="T_511ef_row9_col7" class="data row9 col7" >0.8416</td>
                        <td id="T_511ef_row9_col8" class="data row9 col8" >0.0770</td>
            </tr>
            <tr>
                        <th id="T_511ef_level0_row10" class="row_heading level0 row10" >ada</th>
                        <td id="T_511ef_row10_col0" class="data row10 col0" >Ada Boost Classifier</td>
                        <td id="T_511ef_row10_col1" class="data row10 col1" >0.8489</td>
                        <td id="T_511ef_row10_col2" class="data row10 col2" >0.9378</td>
                        <td id="T_511ef_row10_col3" class="data row10 col3" >0.8378</td>
                        <td id="T_511ef_row10_col4" class="data row10 col4" >0.8721</td>
                        <td id="T_511ef_row10_col5" class="data row10 col5" >0.8440</td>
                        <td id="T_511ef_row10_col6" class="data row10 col6" >0.7652</td>
                        <td id="T_511ef_row10_col7" class="data row10 col7" >0.7782</td>
                        <td id="T_511ef_row10_col8" class="data row10 col8" >0.0320</td>
            </tr>
            <tr>
                        <th id="T_511ef_level0_row11" class="row_heading level0 row11" >knn</th>
                        <td id="T_511ef_row11_col0" class="data row11 col0" >K Neighbors Classifier</td>
                        <td id="T_511ef_row11_col1" class="data row11 col1" >0.7178</td>
                        <td id="T_511ef_row11_col2" class="data row11 col2" >0.8830</td>
                        <td id="T_511ef_row11_col3" class="data row11 col3" >0.6922</td>
                        <td id="T_511ef_row11_col4" class="data row11 col4" >0.7521</td>
                        <td id="T_511ef_row11_col5" class="data row11 col5" >0.7012</td>
                        <td id="T_511ef_row11_col6" class="data row11 col6" >0.5651</td>
                        <td id="T_511ef_row11_col7" class="data row11 col7" >0.5915</td>
                        <td id="T_511ef_row11_col8" class="data row11 col8" >0.0090</td>
            </tr>
            <tr>
                        <th id="T_511ef_level0_row12" class="row_heading level0 row12" >svm</th>
                        <td id="T_511ef_row12_col0" class="data row12 col0" >SVM - Linear Kernel</td>
                        <td id="T_511ef_row12_col1" class="data row12 col1" >0.5978</td>
                        <td id="T_511ef_row12_col2" class="data row12 col2" >0.0000</td>
                        <td id="T_511ef_row12_col3" class="data row12 col3" >0.5678</td>
                        <td id="T_511ef_row12_col4" class="data row12 col4" >0.5420</td>
                        <td id="T_511ef_row12_col5" class="data row12 col5" >0.5152</td>
                        <td id="T_511ef_row12_col6" class="data row12 col6" >0.3702</td>
                        <td id="T_511ef_row12_col7" class="data row12 col7" >0.4458</td>
                        <td id="T_511ef_row12_col8" class="data row12 col8" >0.0070</td>
            </tr>
    </tbody></table>


Showing the best classifier:


```python
print(best)
```

    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=-1, oob_score=False, random_state=6420, verbose=0,
                           warm_start=False)


## Tuning a model

The models used when comparing use their default hyperparameters. We can tune and choose the best hyperparameters for a single model using PyCaret. This is done by running a Random Grid Search on a specific search space. We can, as well, define a custom search grid, but we will not do it at this point. Also, there are many hyperparameters that allow us to choose early stopping, number of iterations, which metric to optimize for, and so on. Tuning returns a similar table as before, but each row now shows the result for each validation fold.

We will tune the K Neighbors Classifier, since it performed poorly with its default parameters. To do so, we first create a model with PyCaret:



```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
tuned_knn = tune_model(KNeighborsClassifier(), n_iter=100)
```


<style  type="text/css" >
#T_9a1e1_row10_col0,#T_9a1e1_row10_col1,#T_9a1e1_row10_col2,#T_9a1e1_row10_col3,#T_9a1e1_row10_col4,#T_9a1e1_row10_col5,#T_9a1e1_row10_col6{
            background:  yellow;
        }</style><table id="T_9a1e1_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >AUC</th>        <th class="col_heading level0 col2" >Recall</th>        <th class="col_heading level0 col3" >Prec.</th>        <th class="col_heading level0 col4" >F1</th>        <th class="col_heading level0 col5" >Kappa</th>        <th class="col_heading level0 col6" >MCC</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_9a1e1_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_9a1e1_row0_col0" class="data row0 col0" >1.0000</td>
                        <td id="T_9a1e1_row0_col1" class="data row0 col1" >1.0000</td>
                        <td id="T_9a1e1_row0_col2" class="data row0 col2" >1.0000</td>
                        <td id="T_9a1e1_row0_col3" class="data row0 col3" >1.0000</td>
                        <td id="T_9a1e1_row0_col4" class="data row0 col4" >1.0000</td>
                        <td id="T_9a1e1_row0_col5" class="data row0 col5" >1.0000</td>
                        <td id="T_9a1e1_row0_col6" class="data row0 col6" >1.0000</td>
            </tr>
            <tr>
                        <th id="T_9a1e1_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_9a1e1_row1_col0" class="data row1 col0" >0.9000</td>
                        <td id="T_9a1e1_row1_col1" class="data row1 col1" >0.9300</td>
                        <td id="T_9a1e1_row1_col2" class="data row1 col2" >0.9333</td>
                        <td id="T_9a1e1_row1_col3" class="data row1 col3" >0.9250</td>
                        <td id="T_9a1e1_row1_col4" class="data row1 col4" >0.9016</td>
                        <td id="T_9a1e1_row1_col5" class="data row1 col5" >0.8438</td>
                        <td id="T_9a1e1_row1_col6" class="data row1 col6" >0.8573</td>
            </tr>
            <tr>
                        <th id="T_9a1e1_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_9a1e1_row2_col0" class="data row2 col0" >0.7000</td>
                        <td id="T_9a1e1_row2_col1" class="data row2 col1" >0.7929</td>
                        <td id="T_9a1e1_row2_col2" class="data row2 col2" >0.6944</td>
                        <td id="T_9a1e1_row2_col3" class="data row2 col3" >0.7400</td>
                        <td id="T_9a1e1_row2_col4" class="data row2 col4" >0.7067</td>
                        <td id="T_9a1e1_row2_col5" class="data row2 col5" >0.5385</td>
                        <td id="T_9a1e1_row2_col6" class="data row2 col6" >0.5471</td>
            </tr>
            <tr>
                        <th id="T_9a1e1_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_9a1e1_row3_col0" class="data row3 col0" >0.8000</td>
                        <td id="T_9a1e1_row3_col1" class="data row3 col1" >0.9083</td>
                        <td id="T_9a1e1_row3_col2" class="data row3 col2" >0.8056</td>
                        <td id="T_9a1e1_row3_col3" class="data row3 col3" >0.8250</td>
                        <td id="T_9a1e1_row3_col4" class="data row3 col4" >0.7971</td>
                        <td id="T_9a1e1_row3_col5" class="data row3 col5" >0.6970</td>
                        <td id="T_9a1e1_row3_col6" class="data row3 col6" >0.7078</td>
            </tr>
            <tr>
                        <th id="T_9a1e1_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_9a1e1_row4_col0" class="data row4 col0" >0.9000</td>
                        <td id="T_9a1e1_row4_col1" class="data row4 col1" >0.9012</td>
                        <td id="T_9a1e1_row4_col2" class="data row4 col2" >0.8889</td>
                        <td id="T_9a1e1_row4_col3" class="data row4 col3" >0.9200</td>
                        <td id="T_9a1e1_row4_col4" class="data row4 col4" >0.8956</td>
                        <td id="T_9a1e1_row4_col5" class="data row4 col5" >0.8462</td>
                        <td id="T_9a1e1_row4_col6" class="data row4 col6" >0.8598</td>
            </tr>
            <tr>
                        <th id="T_9a1e1_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_9a1e1_row5_col0" class="data row5 col0" >0.8000</td>
                        <td id="T_9a1e1_row5_col1" class="data row5 col1" >0.8298</td>
                        <td id="T_9a1e1_row5_col2" class="data row5 col2" >0.7778</td>
                        <td id="T_9a1e1_row5_col3" class="data row5 col3" >0.8450</td>
                        <td id="T_9a1e1_row5_col4" class="data row5 col4" >0.7627</td>
                        <td id="T_9a1e1_row5_col5" class="data row5 col5" >0.6923</td>
                        <td id="T_9a1e1_row5_col6" class="data row5 col6" >0.7273</td>
            </tr>
            <tr>
                        <th id="T_9a1e1_level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_9a1e1_row6_col0" class="data row6 col0" >0.7000</td>
                        <td id="T_9a1e1_row6_col1" class="data row6 col1" >0.8857</td>
                        <td id="T_9a1e1_row6_col2" class="data row6 col2" >0.6944</td>
                        <td id="T_9a1e1_row6_col3" class="data row6 col3" >0.7400</td>
                        <td id="T_9a1e1_row6_col4" class="data row6 col4" >0.7067</td>
                        <td id="T_9a1e1_row6_col5" class="data row6 col5" >0.5385</td>
                        <td id="T_9a1e1_row6_col6" class="data row6 col6" >0.5471</td>
            </tr>
            <tr>
                        <th id="T_9a1e1_level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_9a1e1_row7_col0" class="data row7 col0" >0.9000</td>
                        <td id="T_9a1e1_row7_col1" class="data row7 col1" >1.0000</td>
                        <td id="T_9a1e1_row7_col2" class="data row7 col2" >0.9167</td>
                        <td id="T_9a1e1_row7_col3" class="data row7 col3" >0.9333</td>
                        <td id="T_9a1e1_row7_col4" class="data row7 col4" >0.9029</td>
                        <td id="T_9a1e1_row7_col5" class="data row7 col5" >0.8485</td>
                        <td id="T_9a1e1_row7_col6" class="data row7 col6" >0.8616</td>
            </tr>
            <tr>
                        <th id="T_9a1e1_level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_9a1e1_row8_col0" class="data row8 col0" >0.8000</td>
                        <td id="T_9a1e1_row8_col1" class="data row8 col1" >0.8458</td>
                        <td id="T_9a1e1_row8_col2" class="data row8 col2" >0.8333</td>
                        <td id="T_9a1e1_row8_col3" class="data row8 col3" >0.9000</td>
                        <td id="T_9a1e1_row8_col4" class="data row8 col4" >0.8000</td>
                        <td id="T_9a1e1_row8_col5" class="data row8 col5" >0.7059</td>
                        <td id="T_9a1e1_row8_col6" class="data row8 col6" >0.7500</td>
            </tr>
            <tr>
                        <th id="T_9a1e1_level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_9a1e1_row9_col0" class="data row9 col0" >0.7778</td>
                        <td id="T_9a1e1_row9_col1" class="data row9 col1" >0.9000</td>
                        <td id="T_9a1e1_row9_col2" class="data row9 col2" >0.6667</td>
                        <td id="T_9a1e1_row9_col3" class="data row9 col3" >0.6296</td>
                        <td id="T_9a1e1_row9_col4" class="data row9 col4" >0.6889</td>
                        <td id="T_9a1e1_row9_col5" class="data row9 col5" >0.6250</td>
                        <td id="T_9a1e1_row9_col6" class="data row9 col6" >0.6934</td>
            </tr>
            <tr>
                        <th id="T_9a1e1_level0_row10" class="row_heading level0 row10" >Mean</th>
                        <td id="T_9a1e1_row10_col0" class="data row10 col0" >0.8278</td>
                        <td id="T_9a1e1_row10_col1" class="data row10 col1" >0.8994</td>
                        <td id="T_9a1e1_row10_col2" class="data row10 col2" >0.8211</td>
                        <td id="T_9a1e1_row10_col3" class="data row10 col3" >0.8458</td>
                        <td id="T_9a1e1_row10_col4" class="data row10 col4" >0.8162</td>
                        <td id="T_9a1e1_row10_col5" class="data row10 col5" >0.7335</td>
                        <td id="T_9a1e1_row10_col6" class="data row10 col6" >0.7551</td>
            </tr>
            <tr>
                        <th id="T_9a1e1_level0_row11" class="row_heading level0 row11" >SD</th>
                        <td id="T_9a1e1_row11_col0" class="data row11 col0" >0.0910</td>
                        <td id="T_9a1e1_row11_col1" class="data row11 col1" >0.0637</td>
                        <td id="T_9a1e1_row11_col2" class="data row11 col2" >0.1079</td>
                        <td id="T_9a1e1_row11_col3" class="data row11 col3" >0.1076</td>
                        <td id="T_9a1e1_row11_col4" class="data row11 col4" >0.0993</td>
                        <td id="T_9a1e1_row11_col5" class="data row11 col5" >0.1417</td>
                        <td id="T_9a1e1_row11_col6" class="data row11 col6" >0.1364</td>
            </tr>
    </tbody></table>



```python
print(tuned_knn)
```

    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                         metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                         weights='distance')


## Plotting a model

If just seeing a table is not enough, you can plot the results of a model in different ways. Here are a few examples:


```python
plot_model(best, plot='boundary')
```


    
<img src="/assets/images/PyCaret_Tutorial_files/PyCaret_Tutorial_28_0.png" />



```python
plot_model(best, plot='confusion_matrix')
```


    
<img src="/assets/images/PyCaret_Tutorial_files/PyCaret_Tutorial_29_0.png" />



```python
plot_model(tuned_knn, plot='pr')
```


    
<img src="/assets/images/PyCaret_Tutorial_files/PyCaret_Tutorial_30_0.png" />
    



```python
plot_model(tuned_knn, plot='class_report')
```


    
<img src="/assets/images/PyCaret_Tutorial_files/PyCaret_Tutorial_31_0.png" />
    



```python
plot_model(tuned_knn, plot='confusion_matrix')
```


    
<img src="/assets/images/PyCaret_Tutorial_files/PyCaret_Tutorial_32_0.png" />
    



```python
plot_model(tuned_knn, plot='auc')
```


    
<img src="/assets/images/PyCaret_Tutorial_files/PyCaret_Tutorial_33_0.png" />
    


## Explainable AI

Most businesses do not like to have a black-box model telling them what to do. It is extremely important to understand what the model does so that the business can take action to improve their results.

For this, we need to install the [shap](https://shap.readthedocs.io/en/latest/index.html) library:

```shell
$ pip install shap
```

Or on the notebook:

```notebook
! pip install shap
```

This library is based on the concept of Shapley values created in the Game Theory context to compute feature importance. For now, it is only available for tree-based models.


```python
from sklearn.ensemble import RandomForestClassifier
rf = create_model(RandomForestClassifier())
```


<style  type="text/css" >
#T_2ca70_row10_col0,#T_2ca70_row10_col1,#T_2ca70_row10_col2,#T_2ca70_row10_col3,#T_2ca70_row10_col4,#T_2ca70_row10_col5,#T_2ca70_row10_col6{
            background:  yellow;
        }</style><table id="T_2ca70_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >AUC</th>        <th class="col_heading level0 col2" >Recall</th>        <th class="col_heading level0 col3" >Prec.</th>        <th class="col_heading level0 col4" >F1</th>        <th class="col_heading level0 col5" >Kappa</th>        <th class="col_heading level0 col6" >MCC</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_2ca70_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_2ca70_row0_col0" class="data row0 col0" >1.0000</td>
                        <td id="T_2ca70_row0_col1" class="data row0 col1" >1.0000</td>
                        <td id="T_2ca70_row0_col2" class="data row0 col2" >1.0000</td>
                        <td id="T_2ca70_row0_col3" class="data row0 col3" >1.0000</td>
                        <td id="T_2ca70_row0_col4" class="data row0 col4" >1.0000</td>
                        <td id="T_2ca70_row0_col5" class="data row0 col5" >1.0000</td>
                        <td id="T_2ca70_row0_col6" class="data row0 col6" >1.0000</td>
            </tr>
            <tr>
                        <th id="T_2ca70_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_2ca70_row1_col0" class="data row1 col0" >1.0000</td>
                        <td id="T_2ca70_row1_col1" class="data row1 col1" >1.0000</td>
                        <td id="T_2ca70_row1_col2" class="data row1 col2" >1.0000</td>
                        <td id="T_2ca70_row1_col3" class="data row1 col3" >1.0000</td>
                        <td id="T_2ca70_row1_col4" class="data row1 col4" >1.0000</td>
                        <td id="T_2ca70_row1_col5" class="data row1 col5" >1.0000</td>
                        <td id="T_2ca70_row1_col6" class="data row1 col6" >1.0000</td>
            </tr>
            <tr>
                        <th id="T_2ca70_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_2ca70_row2_col0" class="data row2 col0" >1.0000</td>
                        <td id="T_2ca70_row2_col1" class="data row2 col1" >1.0000</td>
                        <td id="T_2ca70_row2_col2" class="data row2 col2" >1.0000</td>
                        <td id="T_2ca70_row2_col3" class="data row2 col3" >1.0000</td>
                        <td id="T_2ca70_row2_col4" class="data row2 col4" >1.0000</td>
                        <td id="T_2ca70_row2_col5" class="data row2 col5" >1.0000</td>
                        <td id="T_2ca70_row2_col6" class="data row2 col6" >1.0000</td>
            </tr>
            <tr>
                        <th id="T_2ca70_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_2ca70_row3_col0" class="data row3 col0" >0.9000</td>
                        <td id="T_2ca70_row3_col1" class="data row3 col1" >1.0000</td>
                        <td id="T_2ca70_row3_col2" class="data row3 col2" >0.9167</td>
                        <td id="T_2ca70_row3_col3" class="data row3 col3" >0.9250</td>
                        <td id="T_2ca70_row3_col4" class="data row3 col4" >0.9000</td>
                        <td id="T_2ca70_row3_col5" class="data row3 col5" >0.8507</td>
                        <td id="T_2ca70_row3_col6" class="data row3 col6" >0.8636</td>
            </tr>
            <tr>
                        <th id="T_2ca70_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_2ca70_row4_col0" class="data row4 col0" >1.0000</td>
                        <td id="T_2ca70_row4_col1" class="data row4 col1" >1.0000</td>
                        <td id="T_2ca70_row4_col2" class="data row4 col2" >1.0000</td>
                        <td id="T_2ca70_row4_col3" class="data row4 col3" >1.0000</td>
                        <td id="T_2ca70_row4_col4" class="data row4 col4" >1.0000</td>
                        <td id="T_2ca70_row4_col5" class="data row4 col5" >1.0000</td>
                        <td id="T_2ca70_row4_col6" class="data row4 col6" >1.0000</td>
            </tr>
            <tr>
                        <th id="T_2ca70_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_2ca70_row5_col0" class="data row5 col0" >0.9000</td>
                        <td id="T_2ca70_row5_col1" class="data row5 col1" >1.0000</td>
                        <td id="T_2ca70_row5_col2" class="data row5 col2" >0.9167</td>
                        <td id="T_2ca70_row5_col3" class="data row5 col3" >0.9250</td>
                        <td id="T_2ca70_row5_col4" class="data row5 col4" >0.9000</td>
                        <td id="T_2ca70_row5_col5" class="data row5 col5" >0.8507</td>
                        <td id="T_2ca70_row5_col6" class="data row5 col6" >0.8636</td>
            </tr>
            <tr>
                        <th id="T_2ca70_level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_2ca70_row6_col0" class="data row6 col0" >1.0000</td>
                        <td id="T_2ca70_row6_col1" class="data row6 col1" >1.0000</td>
                        <td id="T_2ca70_row6_col2" class="data row6 col2" >1.0000</td>
                        <td id="T_2ca70_row6_col3" class="data row6 col3" >1.0000</td>
                        <td id="T_2ca70_row6_col4" class="data row6 col4" >1.0000</td>
                        <td id="T_2ca70_row6_col5" class="data row6 col5" >1.0000</td>
                        <td id="T_2ca70_row6_col6" class="data row6 col6" >1.0000</td>
            </tr>
            <tr>
                        <th id="T_2ca70_level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_2ca70_row7_col0" class="data row7 col0" >0.9000</td>
                        <td id="T_2ca70_row7_col1" class="data row7 col1" >0.9667</td>
                        <td id="T_2ca70_row7_col2" class="data row7 col2" >0.9167</td>
                        <td id="T_2ca70_row7_col3" class="data row7 col3" >0.9200</td>
                        <td id="T_2ca70_row7_col4" class="data row7 col4" >0.8984</td>
                        <td id="T_2ca70_row7_col5" class="data row7 col5" >0.8438</td>
                        <td id="T_2ca70_row7_col6" class="data row7 col6" >0.8573</td>
            </tr>
            <tr>
                        <th id="T_2ca70_level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_2ca70_row8_col0" class="data row8 col0" >1.0000</td>
                        <td id="T_2ca70_row8_col1" class="data row8 col1" >1.0000</td>
                        <td id="T_2ca70_row8_col2" class="data row8 col2" >1.0000</td>
                        <td id="T_2ca70_row8_col3" class="data row8 col3" >1.0000</td>
                        <td id="T_2ca70_row8_col4" class="data row8 col4" >1.0000</td>
                        <td id="T_2ca70_row8_col5" class="data row8 col5" >1.0000</td>
                        <td id="T_2ca70_row8_col6" class="data row8 col6" >1.0000</td>
            </tr>
            <tr>
                        <th id="T_2ca70_level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_2ca70_row9_col0" class="data row9 col0" >1.0000</td>
                        <td id="T_2ca70_row9_col1" class="data row9 col1" >1.0000</td>
                        <td id="T_2ca70_row9_col2" class="data row9 col2" >1.0000</td>
                        <td id="T_2ca70_row9_col3" class="data row9 col3" >1.0000</td>
                        <td id="T_2ca70_row9_col4" class="data row9 col4" >1.0000</td>
                        <td id="T_2ca70_row9_col5" class="data row9 col5" >1.0000</td>
                        <td id="T_2ca70_row9_col6" class="data row9 col6" >1.0000</td>
            </tr>
            <tr>
                        <th id="T_2ca70_level0_row10" class="row_heading level0 row10" >Mean</th>
                        <td id="T_2ca70_row10_col0" class="data row10 col0" >0.9700</td>
                        <td id="T_2ca70_row10_col1" class="data row10 col1" >0.9967</td>
                        <td id="T_2ca70_row10_col2" class="data row10 col2" >0.9750</td>
                        <td id="T_2ca70_row10_col3" class="data row10 col3" >0.9770</td>
                        <td id="T_2ca70_row10_col4" class="data row10 col4" >0.9698</td>
                        <td id="T_2ca70_row10_col5" class="data row10 col5" >0.9545</td>
                        <td id="T_2ca70_row10_col6" class="data row10 col6" >0.9585</td>
            </tr>
            <tr>
                        <th id="T_2ca70_level0_row11" class="row_heading level0 row11" >SD</th>
                        <td id="T_2ca70_row11_col0" class="data row11 col0" >0.0458</td>
                        <td id="T_2ca70_row11_col1" class="data row11 col1" >0.0100</td>
                        <td id="T_2ca70_row11_col2" class="data row11 col2" >0.0382</td>
                        <td id="T_2ca70_row11_col3" class="data row11 col3" >0.0352</td>
                        <td id="T_2ca70_row11_col4" class="data row11 col4" >0.0461</td>
                        <td id="T_2ca70_row11_col5" class="data row11 col5" >0.0695</td>
                        <td id="T_2ca70_row11_col6" class="data row11 col6" >0.0635</td>
            </tr>
    </tbody></table>


### Summary plot


```python
interpret_model(rf, plot='summary')
```


    
<img src="/assets/images/PyCaret_Tutorial_files/PyCaret_Tutorial_37_0.png" />


### Correlation plot


```python
interpret_model(rf, plot='correlation')
```


    
<img src="/assets/images/PyCaret_Tutorial_files/PyCaret_Tutorial_39_0.png" />
    


### Reason plot

#### All observation


```python
interpret_model(rf, plot='reason')
```


<img src="/assets/images/PyCaret_Tutorial_files/PyCaret_Tutorial_40_0.png" />



#### A specific observation


```python
interpret_model(rf, plot='reason', observation=10)
```


<img src="/assets/images/PyCaret_Tutorial_files/PyCaret_Tutorial_41_0.png" />



## Predicting

Now that the exploratory phase is over, we can predict the results on unseen data.


```python
best_final = finalize_model(best)
predictions = predict_model(best_final, data=df_test.drop('target', axis=1))
```


```python
from sklearn.metrics import classification_report

print(classification_report(df_test['target'], predictions['Label']))
```

                  precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00        12
             1.0       1.00      1.00      1.00        14
             2.0       1.00      1.00      1.00        10
    
        accuracy                           1.00        36
       macro avg       1.00      1.00      1.00        36
    weighted avg       1.00      1.00      1.00        36
    


## What now?

PyCaret allows you to do predictions direct from the cloud, so check it out in their [website](https://pycaret.org/predict-model/) if you want to do this.

Also, you can build your [ensemble](https://pycaret.org/ensemble-model/) models with PyCaret.

They support not only Classification, but also Regression, Anomaly Detection, Clustering, Natural Language Processing, and Association Rules Mining.

##### References

[1] [PyCaret Homepage](https://pycaret.org/)

[2] [PyCaret documentation](https://pycaret.readthedocs.io/)

[3] [Multiclass Classification Tutorial](https://github.com/pycaret/pycaret/blob/master/tutorials/Multiclass%20Classification%20Tutorial%20Level%20Beginner%20-%20MCLF101.ipynb)

[4] [A Gentle Introduction to PyCaret for Machine Learning](https://machinelearningmastery.com/pycaret-for-machine-learning/)
