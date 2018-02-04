---
title: "Compute correlation matrix"
---

It can be cumbersome to get the list of the most correlated pairs of variables in a data set. Here is an example of how to do so, quite smoothly. 

* We first create a toy dataset with 20 features and 5 correlated pairs of features to play with
* Then, the correlation matrix is computed using the `pandas.DataFrame.corr()` command
* To extract the relevant part of the matrix, a boolean mask is created with the `numpy.triu()` command
* Finally, the matrix is converted to a Pandas Series with a multi-index using the `pandas.DataFrame.stack()` command


```python
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

X, y = make_classification(n_features=10, n_informative=3, n_redundant=5, n_classes=2,
    n_clusters_per_class=2)

col_names = ['feature_' + str(i) for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=col_names)
```


```python
# compute correlation matrix
cor_matrix = X.corr()
cor_matrix.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_0</th>
      <th>feature_1</th>
      <th>feature_2</th>
      <th>feature_3</th>
      <th>feature_4</th>
      <th>feature_5</th>
      <th>feature_6</th>
      <th>feature_7</th>
      <th>feature_8</th>
      <th>feature_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>feature_0</th>
      <td>1.000000</td>
      <td>-0.823849</td>
      <td>0.973042</td>
      <td>-0.949994</td>
      <td>-0.896065</td>
      <td>0.764993</td>
      <td>-0.160190</td>
      <td>0.170617</td>
      <td>-0.032467</td>
      <td>0.198784</td>
    </tr>
    <tr>
      <th>feature_1</th>
      <td>-0.823849</td>
      <td>1.000000</td>
      <td>-0.689932</td>
      <td>0.812806</td>
      <td>0.625704</td>
      <td>-0.975139</td>
      <td>0.105197</td>
      <td>-0.631158</td>
      <td>0.000368</td>
      <td>-0.579045</td>
    </tr>
    <tr>
      <th>feature_2</th>
      <td>0.973042</td>
      <td>-0.689932</td>
      <td>1.000000</td>
      <td>-0.877041</td>
      <td>-0.958594</td>
      <td>0.649724</td>
      <td>-0.177422</td>
      <td>0.051843</td>
      <td>-0.046402</td>
      <td>-0.028937</td>
    </tr>
    <tr>
      <th>feature_3</th>
      <td>-0.949994</td>
      <td>0.812806</td>
      <td>-0.877041</td>
      <td>1.000000</td>
      <td>0.718516</td>
      <td>-0.694186</td>
      <td>0.118221</td>
      <td>-0.063240</td>
      <td>0.013887</td>
      <td>-0.428114</td>
    </tr>
    <tr>
      <th>feature_4</th>
      <td>-0.896065</td>
      <td>0.625704</td>
      <td>-0.958594</td>
      <td>0.718516</td>
      <td>1.000000</td>
      <td>-0.648473</td>
      <td>0.193494</td>
      <td>-0.168040</td>
      <td>0.057019</td>
      <td>0.225775</td>
    </tr>
  </tbody>
</table>
</div>



Then we want to extract the upper-part of the matrix (becauses the correlation matrix is symetrical), to do so we will generate a boolean mask array from an upper triangular matrix.


```python
mask = np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool)
print('The following mask has been generated: \n')
print(mask)
```

    The following mask has been generated: 
    
    [[False  True  True  True  True  True  True  True  True  True]
     [False False  True  True  True  True  True  True  True  True]
     [False False False  True  True  True  True  True  True  True]
     [False False False False  True  True  True  True  True  True]
     [False False False False False  True  True  True  True  True]
     [False False False False False False  True  True  True  True]
     [False False False False False False False  True  True  True]
     [False False False False False False False False  True  True]
     [False False False False False False False False False  True]
     [False False False False False False False False False False]]


When applied to our correlation matrix, it will only keep the upper part, excluding the diagonal. We use `.abs()`at the end because we are interested in variables positively and negatively correlated.


```python
upper_cor_matrix = cor_matrix.where(mask).abs()
upper_cor_matrix
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_0</th>
      <th>feature_1</th>
      <th>feature_2</th>
      <th>feature_3</th>
      <th>feature_4</th>
      <th>feature_5</th>
      <th>feature_6</th>
      <th>feature_7</th>
      <th>feature_8</th>
      <th>feature_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>feature_0</th>
      <td>NaN</td>
      <td>0.823849</td>
      <td>0.973042</td>
      <td>0.949994</td>
      <td>0.896065</td>
      <td>0.764993</td>
      <td>0.160190</td>
      <td>0.170617</td>
      <td>0.032467</td>
      <td>0.198784</td>
    </tr>
    <tr>
      <th>feature_1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.689932</td>
      <td>0.812806</td>
      <td>0.625704</td>
      <td>0.975139</td>
      <td>0.105197</td>
      <td>0.631158</td>
      <td>0.000368</td>
      <td>0.579045</td>
    </tr>
    <tr>
      <th>feature_2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.877041</td>
      <td>0.958594</td>
      <td>0.649724</td>
      <td>0.177422</td>
      <td>0.051843</td>
      <td>0.046402</td>
      <td>0.028937</td>
    </tr>
    <tr>
      <th>feature_3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.718516</td>
      <td>0.694186</td>
      <td>0.118221</td>
      <td>0.063240</td>
      <td>0.013887</td>
      <td>0.428114</td>
    </tr>
    <tr>
      <th>feature_4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.648473</td>
      <td>0.193494</td>
      <td>0.168040</td>
      <td>0.057019</td>
      <td>0.225775</td>
    </tr>
    <tr>
      <th>feature_5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.115363</td>
      <td>0.756552</td>
      <td>0.006447</td>
      <td>0.460539</td>
    </tr>
    <tr>
      <th>feature_6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.034566</td>
      <td>0.064570</td>
      <td>0.069284</td>
    </tr>
    <tr>
      <th>feature_7</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.012584</td>
      <td>0.361545</td>
    </tr>
    <tr>
      <th>feature_8</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.058279</td>
    </tr>
    <tr>
      <th>feature_9</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



To make it easier to use, the columns are stacked into rows, resulting in a multi-index Pandas Series:


```python
cor_series = upper_cor_matrix.stack().sort_values(ascending=False)
print('Display the most-correlated pairs:')
cor_series[cor_series > 0.6]
```

    Display the most-correlated pairs:





    feature_1  feature_5    0.975139
    feature_0  feature_2    0.973042
    feature_2  feature_4    0.958594
    feature_0  feature_3    0.949994
               feature_4    0.896065
    feature_2  feature_3    0.877041
    feature_0  feature_1    0.823849
    feature_1  feature_3    0.812806
    feature_0  feature_5    0.764993
    feature_5  feature_7    0.756552
    feature_3  feature_4    0.718516
               feature_5    0.694186
    feature_1  feature_2    0.689932
    feature_2  feature_5    0.649724
    feature_4  feature_5    0.648473
    feature_1  feature_7    0.631158
               feature_4    0.625704
    dtype: float64


