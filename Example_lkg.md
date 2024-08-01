# Example

The following example demonstrates how to the use the `pymatch` package to match [Lending Club Loan Data](https://www.kaggle.com/datasets/imsparsh/lending-club-loan-dataset-2007-2011). Follow the link to download the dataset from Kaggle (you'll have to create an account, it's fast and free!).

Lending Club是最大的在线贷款市场，提供个人贷款、商业贷款和医疗程序融资。借款人可以通过快速的在线界面轻松获得较低利率的贷款。

这些文件包含 2007 年至 2011 年发放的所有贷款的完整贷款数据，包括当前贷款状态（当前、已注销、已全额支付等）和最新付款信息。包含截至“现在”的贷款数据的文件包含截至上一个完整日历季度发放的所有贷款的完整贷款数据。其他功能包括信用评分、财务查询次数和收款等。该文件是一个包含约 39,000 个观测值和 111 个变量的矩阵。数据字典在单独的文件中提供。


Here we match Lending Club users that fully paid off loans (control) to those that defaulted (test). The example is contrived, however a use case for this could be that we want to analyze user sentiment with the platform. Users that default on loans may have worse sentiment because they are predisposed to a bad situation--influencing their perception of the product. Before analyzing sentiment, we can match users that paid their loans in full to users that defaulted based on the characteristics we can observe. If matching is successful, we could then make a statetment about the **causal effect** defaulting has on sentiment if we are confident our samples are sufficiently balanced and our model is free from omitted variable bias.

在这里，我们将完全还清贷款的 Lending Club 用户（对照组）与违约用户（测试组）进行匹配。
这个例子是人为的，但是一个用例可能是我们想要分析平台上的用户情绪。
违约贷款的用户可能会有更糟糕的情绪，因为他们容易陷入糟糕的境地——这会影响他们对产品的看法。
在分析情绪之前，我们可以根据我们观察到的特征将全额偿还贷款的用户与违约用户进行匹配。
如果匹配成功，我们就可以对违约对情绪的因果影响做出陈述，前提是我们相信我们的样本足够平衡，并且我们的模型没有遗漏变量偏差。

This example, however, only goes through the matching procedure, which can be broken down into the following steps:

* [Data Preparation](#Data-Prep)
* [Fit Propensity Score Models](#Matcher)
* [Predict Propensity Scores](#Predict-Scores)
* [Tune Threshold](#Tune-Threshold)
* [Match Data](#Match-Data)
* [Assess Matches](#Assess-Matches)

----

### Data Prep


```python
%load_ext autoreload
%autoreload 2
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
from psmatch.Matcher import Matcher
import pandas as pd
import numpy as np
```

Load the dataset (`loan.csv`), which is the original `loan.csv` sampled as follows:

```python
path = 'misc/loan_full.csv'
fields = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "loan_status"
]
data = pd.read_csv(path)[fields]

# Treat long late as Defaulted
data.loc[data['loan_status'] == 'Late (31-120 days)', 'loan_status'] = 'Default'

# Sample 20K records from Fully paid and 2K from default
df = data[data.loan_status == 'Fully Paid'].sample(20000, random_state=42) \
    .append(data[data.loan_status == 'Default'].sample(2000, random_state=42))

df.to_csv('misc/loan.csv', index=False)
```

loan_amnt 贷款金额： 借款人申请的贷款金额。如果在某个时间点信用部门减少了贷款金额，则该值会反映出来。

funded_amnt 贷款金额：在那个时间点上对应贷款的承诺总金额。

funded_amnt_inv 投资者已出资金额：在那个时间点上投资者为该贷款承诺的总金额。

grade 贷款评级：LC分配的贷款等级。

sub_grade LC 分配的贷款子等级

installment 借款人每月应偿还的付款额（如果贷款开始）。

int_rate 贷款利率。

term	The number of payments on the loan. Values are in months and can be either 36 or 60.

干预点

loan_status 贷款状态： 当前贷款的状态



```python
path = 'misc/loan_full.csv'
fields = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "loan_status"
]
data = pd.read_csv(path)[fields]

# Treat long late as Defaulted
# data.loc[data['loan_status'] == 'Late (31-120 days)', 'loan_status'] = 'Default'

data.loc[data['loan_status'] == 'Charged Off', 'loan_status'] = 'Default'


# Sample 20K records from Fully paid and 2K from default
df = data[data.loan_status == 'Fully Paid'].sample(20000, random_state=42) \
    .append(data[data.loan_status == 'Default'].sample(2000, random_state=42))

# df = data[data.loan_status == 'Fully Paid'].sample(20000, random_state=42) \
#     .append(data[data.loan_status == 'Charged Off'].sample(2000, random_state=42))


# df.to_csv('misc/loan.csv', index=False)
# df.to_csv('misc/loan_large.csv', index=False)
```


```python
data.dtypes
```




    loan_amnt            int64
    funded_amnt          int64
    funded_amnt_inv    float64
    term                object
    int_rate           float64
    installment        float64
    grade               object
    sub_grade           object
    loan_status         object
    dtype: object




```python
# Visual Python: Data Analysis > Groupby
data.groupby('loan_status').count()
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
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Default</th>
      <td>2000</td>
      <td>2000</td>
      <td>2000</td>
      <td>2000</td>
      <td>2000</td>
      <td>2000</td>
      <td>2000</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>20000</td>
      <td>20000</td>
      <td>20000</td>
      <td>20000</td>
      <td>20000</td>
      <td>20000</td>
      <td>20000</td>
      <td>20000</td>
    </tr>
  </tbody>
</table>
</div>




```python
path = "misc/loan.csv"
path = 'misc/loan_large.csv'
data = pd.read_csv(path)
data['int_rate'] = data['int_rate'].str.rstrip('%').astype('float') 
```


```python
data
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
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>loan_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13000</td>
      <td>13000</td>
      <td>13000.000000</td>
      <td>36 months</td>
      <td>7.14</td>
      <td>402.24</td>
      <td>A</td>
      <td>A3</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6400</td>
      <td>6400</td>
      <td>6400.000000</td>
      <td>36 months</td>
      <td>10.65</td>
      <td>208.47</td>
      <td>B</td>
      <td>B2</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>1000</td>
      <td>1000.000000</td>
      <td>36 months</td>
      <td>13.23</td>
      <td>33.81</td>
      <td>C</td>
      <td>C1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17500</td>
      <td>17500</td>
      <td>17500.000000</td>
      <td>60 months</td>
      <td>11.71</td>
      <td>386.72</td>
      <td>B</td>
      <td>B3</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5000</td>
      <td>5000</td>
      <td>5000.000000</td>
      <td>36 months</td>
      <td>6.03</td>
      <td>152.18</td>
      <td>A</td>
      <td>A1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21995</th>
      <td>10000</td>
      <td>10000</td>
      <td>10000.000000</td>
      <td>36 months</td>
      <td>13.48</td>
      <td>339.25</td>
      <td>C</td>
      <td>C3</td>
      <td>Default</td>
    </tr>
    <tr>
      <th>21996</th>
      <td>1000</td>
      <td>1000</td>
      <td>1000.000000</td>
      <td>36 months</td>
      <td>19.03</td>
      <td>36.68</td>
      <td>E</td>
      <td>E2</td>
      <td>Default</td>
    </tr>
    <tr>
      <th>21997</th>
      <td>2400</td>
      <td>2400</td>
      <td>2300.000000</td>
      <td>36 months</td>
      <td>13.92</td>
      <td>81.93</td>
      <td>C</td>
      <td>C4</td>
      <td>Default</td>
    </tr>
    <tr>
      <th>21998</th>
      <td>25000</td>
      <td>25000</td>
      <td>23700.000000</td>
      <td>36 months</td>
      <td>14.83</td>
      <td>864.56</td>
      <td>D</td>
      <td>D3</td>
      <td>Default</td>
    </tr>
    <tr>
      <th>21999</th>
      <td>25000</td>
      <td>25000</td>
      <td>6100.000502</td>
      <td>60 months</td>
      <td>13.43</td>
      <td>574.35</td>
      <td>C</td>
      <td>C3</td>
      <td>Default</td>
    </tr>
  </tbody>
</table>
<p>22000 rows × 9 columns</p>
</div>



Create test and control groups and reassign `loan_status` to be a binary treatment indicator. This is our reponse in the logistic regression model(s) used to generate propensity scores.


```python
test = data[data.loan_status == "Default"]
control = data[data.loan_status == "Fully Paid"]
test['loan_status'] = 1
control['loan_status'] = 0
```

----

### `Matcher`

Initalize the `Matcher` object. 

**Note that:**

* Upon intialization, `Matcher` prints the formula used to fit logistic regression model(s) and the number of records in the majority/minority class. 
    * The regression model(s) are used to generate propensity scores. In this case, we are using the covariates on the right side of the equation to estimate the probability of defaulting on a loan (`loan_status`= 1). 
* `Matcher` will use all covariates in the dataset unless a formula is specified by the user. Note that this step is only fitting model(s), we assign propensity scores later. 
* Any covariates passed to the (optional) `exclude` parameter will be ignored from the model fitting process. This parameter is particularly useful for unique identifiers like a `user_id`. 


```python
m = Matcher(test, control, yvar="loan_status", exclude=[])
```

    Formula:
    loan_status ~ funded_amnt+funded_amnt_inv+grade+installment+int_rate+loan_amnt+sub_grade+term
    1 0
    n majority: 20000
    n minority: 2000


There is a significant imbalance in our data--the majority group (fully-paid loans) having many more records than the minority group (defaulted loans). We account for this by setting `balance=True` when calling `Matcher.fit_scores()` below. This tells `Matcher` to sample from the majority group when fitting the logistic regression model(s) so that the groups are of equal size. When undersampling this way, it is highly recommended that `nmodels` is explictly assigned to a integer much larger than 1. This ensure is that more of the majority group is contributing to the generation of propensity scores. The value of this integer should depend on the severity of the imbalance; here we use `nmodels`=100.


```python
# for reproducibility
np.random.seed(20170925)

# m.fit_scores(balance=True, nmodels=10)
```

The average accuracy of our 100 models is 66.06%, suggesting that there's separability within our data and justifiying the need for the matching procedure. It's worth noting that we don't pay much attention to these logistic models since we are using them as a feature extraction tool (generation of propensity scores). The accuracy is a good way to detect separability at a glance, but we shouldn't spend time tuning and tinkering with these models. If our accuracy was close to 50%, that would suggest we cannot detect much separability in our groups given the features we observe and that matching is probably not necessary (or more features should be included if possible).


我们100个模型的平均准确率是66.06%，这表明我们的数据中存在可分性，并且证明了匹配程序的必要性。值得注意的是，我们不太关注这些逻辑回归模型，因为我们只是将它们用作特征提取工具（生成倾向得分）。准确率是快速检测可分性的一个好方法，但我们不应该花时间调整和修补这些模型。如果我们的准确率接近50%，那就表明，鉴于我们观察到的特征，我们的分组中几乎没有可分性，可能不需要进行匹配（或者如果可能的话，应该包含更多特征）。

为什么在倾向性得分匹配模型,不需要花时间调整和修补这些模型? 为什么说平均准确率是66.06%，这表明我们的数据中存在可分性，并且证明了匹配程序的必要性。

```sql
在倾向性得分匹配（Propensity Score Matching，简称PSM）模型中，进行调整和修补的需求较少，主要是由于以下几个原因：

匹配过程：PSM的核心是根据倾向性得分（即一个观测个体被处理组的概率）进行匹配。这个过程通常包括选择合适的匹配方法（如一对一匹配、卡尺匹配、最近邻匹配等）和验证匹配的质量。这个步骤本质上是数据的预处理，与调整传统的机器学习模型参数（如权重调整、正则化等）是不同的。

模型简单性：倾向性得分主要由逻辑回归或类似简单的回归模型计算得出。这些模型本身通常较为简单，参数较少，因此在实践中很少需要复杂的调整。重点是确保平衡，而不是优化预测准确度。

衡量匹配效果：匹配程序的成功与否主要通过平衡性检验来验证，即检查匹配后处理组和对照组在协变量上的相似性。这种平衡性是整个方法的核心，所以一旦平衡性符合要求，后续分析的可靠性基本上得到保障，不需要进一步调整模型。

关于“平均准确率是66.06%，这表明我们的数据中存在可分性，并且证明了匹配程序的必要性”的解释：

平均准确率：在二分类问题中，如果一个模型的预测准确率显著高于50%，则表示该模型在某种程度上能够区分两类样本。在这里，66.06%的平均准确率虽然不是很高，但已经显著高于随机猜测（50%），表明在数据中存在一定的模式和差异，这些模式和差异可以用来区分处理组和对照组。

可分性：数据的可分性指的是处理组和对照组在特征空间上的分离程度。如果模型可以达到66.06%的准确率，意味着处理组和对照组在特征空间上存在差异。

匹配程序的必要性：在倾向性得分匹配中，我们希望通过匹配将处理组和对照组在协变量上的分布调整得尽可能相似（即平衡）。平均准确率为66.06%说明存在一个倾向性得分模型，该模型能够以一定的准确性预测处理组与对照组之间的差异。如果没有匹配，用这个不平衡的数据直接进行处理效果评估可能会产生偏差。因此，匹配程序通过平衡协变量分布，帮助我们减少潜在的偏倚，提高因果推断的可靠性。

总结一下，高于随机水平的平均准确率，表明存在潜在的协变量差异，而通过PSM，可以使处理组和对照组在这些协变量上更加平衡，从而提高结果分析的准确性和可信度。
```



```python
np.random.seed(20210419)

m.fit_scores(balance=True, nmodels=10,n_jobs = 12,model_type='tree')
```

    This computer has: 12 cores , The workers should be :12
    Fitting Models on Balanced Samples , model number :0
    Fitting Models on Balanced Samples , model number :1
    Fitting Models on Balanced Samples , model number :2
    Fitting Models on Balanced Samples , model number :3
    Fitting Models on Balanced Samples , model number :4
    Fitting Models on Balanced Samples , model number :5
    Fitting Models on Balanced Samples , model number :6
    Fitting Models on Balanced Samples , model number :7
    Fitting Models on Balanced Samples , model number :8
    Fitting Models on Balanced Samples , model number :9
    
    Average Accuracy: 67.93%


### Predict Scores


```python
m.predict_scores()
```


```python
m.plot_scores()
```


    
![png](Example_lkg_files/Example_lkg_21_0.png)
    


The plot above demonstrates the separability present in our data. Test profiles have a much higher **propensity**, or estimated probability of defaulting given the features we isolated in the data.

---

### Tune Threshold

The `Matcher.match()` method matches profiles that have propensity scores within some threshold. 

i.e. for two scores `s1` and `s2`, `|s1 - s2|` <= `threshold`

By default matches are found *from* the majority group *for* the minority group. For example, if our test group contains 1,000 records and our control group contains 20,000, `Matcher` will
    iterate through the test (minority) group and find suitable matches from the control (majority) group. If a record in the minority group has no suitable matches, it is dropped from the final matched dataset. We need to ensure our threshold is small enough such that we get close matches and retain most (or all) of our data in the minority group.
    
Below we tune the threshold using `method="random"`. This matches a random profile that is within the threshold
as there could be many. This is much faster than the alternative method "min", which finds the *closest* match for every minority record.


```python
m.tune_threshold(method='random')
```

    0 / 10 - 0.0 finished 0.2975
    1 / 10 - 0.0001 finished 0.9435
    2 / 10 - 0.0002 finished 0.9735
    3 / 10 - 0.00030000000000000003 finished 0.9805
    4 / 10 - 0.0004 finished 0.985
    5 / 10 - 0.0005 finished 0.9875
    6 / 10 - 0.0006000000000000001 finished 0.9875
    7 / 10 - 0.0007 finished 0.989
    8 / 10 - 0.0008 finished 0.9925
    9 / 10 - 0.0009000000000000001 finished 0.9935



    
![png](Example_lkg_files/Example_lkg_25_1.png)
    


It looks like a threshold of 0.0005 retains enough information in our data. Let's proceed with matching using this threshold.

---

### Match Data

Below we match one record from the majority group to each record in the minority group. This is done **with** replacement, meaning a single majority record can be matched to multiple minority records. `Matcher` assigns a unique `record_id` to each record in the test and control groups so this can be addressed after matching. If susequent modelling is planned, one might consider weighting models using a weight vector of 1/`f` for each record, `f` being a record's frequency in the matched dataset. Thankfully `Matcher` can handle all of this for you :).


```python
# m.match(method="min", nmatches=1, threshold=0.0005)
# m.match(method="random", nmatches=1, threshold=0.01)
# 基本上都能通过p值检验
# m.match(method="random", nmatches=1, threshold=0.00099)
# m.match(method="random", nmatches=1, threshold=0.0001)

# 连续值基本上都通不过
# m.match(method="random", nmatches=1, threshold=0.005)
# 连续值基本上都通不过
# m.match(method="random", nmatches=1, threshold=0.003)

# 基本上都能通过p值检验 - 接近
# m.match(method="random", nmatches=1, threshold=0.0009)

# 基本上都能通过p值检验-基本上不能
# m.match(method="random", nmatches=1, threshold=0.0015)

# m.match(method="min", nmatches=1, threshold=0.005)

# 最小匹配,居然不如随机匹配
m.match(method="min", nmatches=1, threshold=0.001)
```


```python
m.record_frequency()
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
      <th>freq</th>
      <th>n_records</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3470</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>212</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



It looks like the bulk of our matched-majority-group records occur only once, 68 occur twice, ... etc. We can preemptively generate a weight vector using `Matcher.assign_weight_vector()`


```python
m.assign_weight_vector()
```

Let's take a look at our matched data thus far. Note that in addition to the weight vector, `Matcher` has also assigned a `match_id` to each record indicating our (in this cased) *paired* matches since we use `nmatches=1`. We can verify that matched records have `scores` within 0.0001 of each other. 


```python
m.matched_data.sort_values("match_id").head(20)
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
      <th>record_id</th>
      <th>matched_count</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>grade</th>
      <th>installment</th>
      <th>int_rate</th>
      <th>loan_amnt</th>
      <th>loan_status</th>
      <th>sub_grade</th>
      <th>term</th>
      <th>scores</th>
      <th>match_id</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2500</td>
      <td>2500.00000</td>
      <td>C</td>
      <td>85.72</td>
      <td>14.22</td>
      <td>2500</td>
      <td>1</td>
      <td>C5</td>
      <td>36 months</td>
      <td>0.572430</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3014</th>
      <td>11994</td>
      <td>1</td>
      <td>2500</td>
      <td>2500.00000</td>
      <td>C</td>
      <td>85.72</td>
      <td>14.22</td>
      <td>2500</td>
      <td>0</td>
      <td>C5</td>
      <td>36 months</td>
      <td>0.572430</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>20000</td>
      <td>19950.00000</td>
      <td>E</td>
      <td>512.13</td>
      <td>18.39</td>
      <td>20000</td>
      <td>1</td>
      <td>E2</td>
      <td>60 months</td>
      <td>0.716869</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2493</th>
      <td>6707</td>
      <td>2</td>
      <td>15000</td>
      <td>14925.00000</td>
      <td>E</td>
      <td>380.34</td>
      <td>17.93</td>
      <td>15000</td>
      <td>0</td>
      <td>E5</td>
      <td>60 months</td>
      <td>0.716865</td>
      <td>1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>5000</td>
      <td>5000.00000</td>
      <td>A</td>
      <td>155.56</td>
      <td>7.51</td>
      <td>5000</td>
      <td>1</td>
      <td>A3</td>
      <td>36 months</td>
      <td>0.278596</td>
      <td>2</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2853</th>
      <td>10302</td>
      <td>1</td>
      <td>5000</td>
      <td>5000.00000</td>
      <td>A</td>
      <td>155.56</td>
      <td>7.51</td>
      <td>5000</td>
      <td>0</td>
      <td>A3</td>
      <td>36 months</td>
      <td>0.278596</td>
      <td>2</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>5000</td>
      <td>5000.00000</td>
      <td>D</td>
      <td>122.90</td>
      <td>16.49</td>
      <td>5000</td>
      <td>1</td>
      <td>D3</td>
      <td>60 months</td>
      <td>0.746045</td>
      <td>3</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3406</th>
      <td>15872</td>
      <td>3</td>
      <td>4000</td>
      <td>4000.00000</td>
      <td>E</td>
      <td>98.92</td>
      <td>16.77</td>
      <td>4000</td>
      <td>0</td>
      <td>E2</td>
      <td>60 months</td>
      <td>0.746065</td>
      <td>3</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>7000</td>
      <td>7000.00000</td>
      <td>C</td>
      <td>238.73</td>
      <td>13.85</td>
      <td>7000</td>
      <td>1</td>
      <td>C4</td>
      <td>36 months</td>
      <td>0.495698</td>
      <td>4</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2654</th>
      <td>8263</td>
      <td>3</td>
      <td>7000</td>
      <td>7000.00000</td>
      <td>C</td>
      <td>238.73</td>
      <td>13.85</td>
      <td>7000</td>
      <td>0</td>
      <td>C4</td>
      <td>36 months</td>
      <td>0.495698</td>
      <td>4</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>3730</th>
      <td>19336</td>
      <td>2</td>
      <td>15000</td>
      <td>15000.00000</td>
      <td>D</td>
      <td>371.91</td>
      <td>16.89</td>
      <td>15000</td>
      <td>0</td>
      <td>D4</td>
      <td>60 months</td>
      <td>0.741647</td>
      <td>5</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>1</td>
      <td>15000</td>
      <td>15000.00000</td>
      <td>D</td>
      <td>371.91</td>
      <td>16.89</td>
      <td>15000</td>
      <td>1</td>
      <td>D4</td>
      <td>60 months</td>
      <td>0.741647</td>
      <td>5</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1</td>
      <td>7000</td>
      <td>6975.00000</td>
      <td>B</td>
      <td>152.17</td>
      <td>10.99</td>
      <td>7000</td>
      <td>1</td>
      <td>B3</td>
      <td>60 months</td>
      <td>0.611821</td>
      <td>6</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2249</th>
      <td>4481</td>
      <td>1</td>
      <td>9500</td>
      <td>9500.00000</td>
      <td>C</td>
      <td>214.61</td>
      <td>12.68</td>
      <td>9500</td>
      <td>0</td>
      <td>C1</td>
      <td>60 months</td>
      <td>0.611788</td>
      <td>6</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3227</th>
      <td>14253</td>
      <td>1</td>
      <td>6000</td>
      <td>6000.00000</td>
      <td>B</td>
      <td>127.49</td>
      <td>10.00</td>
      <td>6000</td>
      <td>0</td>
      <td>B2</td>
      <td>60 months</td>
      <td>0.551327</td>
      <td>7</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>1</td>
      <td>6000</td>
      <td>6000.00000</td>
      <td>B</td>
      <td>127.49</td>
      <td>10.00</td>
      <td>6000</td>
      <td>1</td>
      <td>B2</td>
      <td>60 months</td>
      <td>0.551327</td>
      <td>7</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>1</td>
      <td>25000</td>
      <td>24975.00000</td>
      <td>E</td>
      <td>658.05</td>
      <td>19.69</td>
      <td>25000</td>
      <td>1</td>
      <td>E5</td>
      <td>60 months</td>
      <td>0.653976</td>
      <td>8</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3563</th>
      <td>17535</td>
      <td>2</td>
      <td>24000</td>
      <td>21455.19459</td>
      <td>C</td>
      <td>573.86</td>
      <td>15.23</td>
      <td>24000</td>
      <td>0</td>
      <td>C5</td>
      <td>60 months</td>
      <td>0.653980</td>
      <td>8</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2868</th>
      <td>10498</td>
      <td>1</td>
      <td>7775</td>
      <td>7775.00000</td>
      <td>C</td>
      <td>185.91</td>
      <td>15.23</td>
      <td>12000</td>
      <td>0</td>
      <td>C5</td>
      <td>60 months</td>
      <td>0.698980</td>
      <td>9</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>1</td>
      <td>5200</td>
      <td>5200.00000</td>
      <td>D</td>
      <td>123.25</td>
      <td>14.83</td>
      <td>5200</td>
      <td>1</td>
      <td>D3</td>
      <td>60 months</td>
      <td>0.698908</td>
      <td>9</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 匹配后的倾向性得分直方图
m.plot_scores('after')
```


    
![png](Example_lkg_files/Example_lkg_35_0.png)
    


---

### Assess Matches

We must now determine if our data is "balanced". Can we detect any statistical differences between the covariates of our matched test and control groups? `Matcher` is configured to treat categorical and continouous variables separately in this assessment.

___Discrete___

For categorical variables, we look at plots comparing the proportional differences between test and control before and after matching. 

For example, the first plot shows:
* `prop_test` - `prop_control` for all possible `term` values---`prop_test` and `prop_control` being the proportion of test and control records with a given term value, respectively. We want these (orange) bars to be small after matching.
* Results (pvalue) of a Chi-Square Test for Independence before and after matching. After matching we want this pvalue to be > 0.05, resulting in our failure to reject the null hypothesis that the frequecy of the enumerated term values are independent of our test and control groups.


```python
# 倾向性得分前后匹配后 - 离散特征的卡方分布变化
categorical_results = m.compare_categorical(return_table=True)
```


    
![png](Example_lkg_files/Example_lkg_39_0.png)
    



    
![png](Example_lkg_files/Example_lkg_39_1.png)
    



    
![png](Example_lkg_files/Example_lkg_39_2.png)
    



```python
categorical_results['是否通过'] = categorical_results['after'] > 0.05
categorical_results
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
      <th>var</th>
      <th>before</th>
      <th>after</th>
      <th>是否通过</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>grade</td>
      <td>0.0</td>
      <td>0.082471</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub_grade</td>
      <td>0.0</td>
      <td>0.646444</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>term</td>
      <td>0.0</td>
      <td>0.055997</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Looking at the plots and test results, we did a pretty good job balancing our categorical features! The p-values from the Chi-Square tests are all > 0.05 and we can verify by observing the small proportional differences in the plots.

___Continuous___

For continous variables we look at Empirical Cumulative Distribution Functions (ECDF) for our test and control groups  before and after matching.

For example, the first plot pair shows:
* ECDF for test vs ECDF for control before matching (left), ECDF for test vs ECDF for control after matching(right). We want the two lines to be very close to each other (or indistiguishable) after matching.
* Some tests + metrics are included in the chart titles.
    * Tests performed:
        * Kolmogorov-Smirnov Goodness of fit Test (KS-test)
            This test statistic is calculated on 1000
            permuted samples of the data, generating
            an imperical p-value.  See pymatch.functions.ks_boot()
            This is an adaptation of the ks.boot() method in 
            the R "Matching" package
            https://www.rdocumentation.org/packages/Matching/versions/4.9-2/topics/ks.boot
        * Chi-Square Distance:
            Similarly this distance metric is calculated on 
            1000 permuted samples. 
            See pymatch.functions.grouped_permutation_test()

    * Other included Stats:
        * Standarized mean and median differences.
             How many standard deviations away are the mean/median
            between our groups before and after matching
            i.e. `abs(mean(control) - mean(test))` / `std(control.union(test))`


```python
cc = m.compare_continuous(return_table=True)
```


    
![png](Example_lkg_files/Example_lkg_42_0.png)
    



    
![png](Example_lkg_files/Example_lkg_42_1.png)
    



    
![png](Example_lkg_files/Example_lkg_42_2.png)
    



    
![png](Example_lkg_files/Example_lkg_42_3.png)
    



    
![png](Example_lkg_files/Example_lkg_42_4.png)
    



```python
cc
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
      <th>var</th>
      <th>ks_before</th>
      <th>ks_after</th>
      <th>grouped_chisqr_before</th>
      <th>grouped_chisqr_after</th>
      <th>std_median_diff_before</th>
      <th>std_median_diff_after</th>
      <th>std_mean_diff_before</th>
      <th>std_mean_diff_after</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>funded_amnt</td>
      <td>0.000</td>
      <td>0.214</td>
      <td>0.001</td>
      <td>0.003</td>
      <td>0.096281</td>
      <td>0.000000</td>
      <td>0.138489</td>
      <td>0.038219</td>
    </tr>
    <tr>
      <th>1</th>
      <td>funded_amnt_inv</td>
      <td>0.000</td>
      <td>0.470</td>
      <td>0.000</td>
      <td>0.004</td>
      <td>0.130111</td>
      <td>-0.001350</td>
      <td>0.103749</td>
      <td>0.012767</td>
    </tr>
    <tr>
      <th>2</th>
      <td>installment</td>
      <td>0.002</td>
      <td>0.056</td>
      <td>0.005</td>
      <td>0.886</td>
      <td>0.084730</td>
      <td>0.073076</td>
      <td>0.056982</td>
      <td>0.054889</td>
    </tr>
    <tr>
      <th>3</th>
      <td>int_rate</td>
      <td>0.000</td>
      <td>0.029</td>
      <td>0.000</td>
      <td>0.011</td>
      <td>0.629946</td>
      <td>-0.099341</td>
      <td>0.634255</td>
      <td>-0.090959</td>
    </tr>
    <tr>
      <th>4</th>
      <td>loan_amnt</td>
      <td>0.000</td>
      <td>0.080</td>
      <td>0.000</td>
      <td>0.116</td>
      <td>0.055038</td>
      <td>0.000000</td>
      <td>0.148540</td>
      <td>0.049170</td>
    </tr>
  </tbody>
</table>
</div>




```python
cc['是否通过'] = (cc['ks_after'] > 0.05) | (cc['grouped_chisqr_after'] > 0.05)
```


```python
cc
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
      <th>var</th>
      <th>ks_before</th>
      <th>ks_after</th>
      <th>grouped_chisqr_before</th>
      <th>grouped_chisqr_after</th>
      <th>std_median_diff_before</th>
      <th>std_median_diff_after</th>
      <th>std_mean_diff_before</th>
      <th>std_mean_diff_after</th>
      <th>是否通过</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>funded_amnt</td>
      <td>0.000</td>
      <td>0.214</td>
      <td>0.001</td>
      <td>0.003</td>
      <td>0.096281</td>
      <td>0.000000</td>
      <td>0.138489</td>
      <td>0.038219</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>funded_amnt_inv</td>
      <td>0.000</td>
      <td>0.470</td>
      <td>0.000</td>
      <td>0.004</td>
      <td>0.130111</td>
      <td>-0.001350</td>
      <td>0.103749</td>
      <td>0.012767</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>installment</td>
      <td>0.002</td>
      <td>0.056</td>
      <td>0.005</td>
      <td>0.886</td>
      <td>0.084730</td>
      <td>0.073076</td>
      <td>0.056982</td>
      <td>0.054889</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>int_rate</td>
      <td>0.000</td>
      <td>0.029</td>
      <td>0.000</td>
      <td>0.011</td>
      <td>0.629946</td>
      <td>-0.099341</td>
      <td>0.634255</td>
      <td>-0.090959</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>loan_amnt</td>
      <td>0.000</td>
      <td>0.080</td>
      <td>0.000</td>
      <td>0.116</td>
      <td>0.055038</td>
      <td>0.000000</td>
      <td>0.148540</td>
      <td>0.049170</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



We want the pvalues from both the KS-test and the grouped permutation of the Chi-Square distance after matching to be > 0.05, and they all are! We can verify by looking at how close the ECDFs are between test and control.

# Conclusion

We saw a very "clean" result from the above procedure, achieving balance among all the covariates. In my work at Mozilla, we see much hairier results using the same procedure, which will likely be your experience too. In the case that certain covariates are not well balanced, one might consider tinkering with the parameters of the matching process (`nmatches`>1) or adding more covariates to the formula specified when we initialized the `Matcher` object.
In any case, in subsequent modelling, you can always control for variables that you haven't deemed "balanced".
