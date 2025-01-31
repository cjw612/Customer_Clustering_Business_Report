# Customer Segmentation Business Analysis Report
https://archive.ics.uci.edu/dataset/352/online+retail

- ### Project Summary
  - #### Project Objective:
    This project aims to compile a business analysis report through segmentation analysis of an online retail dataset.
  - #### Project Overview:
    This project first deploys classical EDA methods to provide an overview of the dataset. Subsequently, RFM (Recency, Frequency, Monetary) indexes are created, which perform as the features used for segmentation. In addition, cancelation rate and customer activity indexes are also created to provide additional information on consumers' characteristics. A comprehensive business analysis report providing next-step suggestions is derived from the results of segmentation analysis.

- ### Data Source
  The dataset used for this analysis is the **"Oline Retail" Dataset (retail.xlsx)** dataset created by Daqing Chen and downloaded from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail).

- ### Data Structure
  This dataset contains 541,909 entries, with a grain of one transaction per item per customer, and eight columns, with each column representing features related to that particular news entry. A snapshot of the dataset is depicted in the following table.
  | InvoiceNo | StockCode | Description                             | Quantity | InvoiceDate          | UnitPrice | CustomerID | Country         |
  |-----------|----------|-----------------------------------------|----------|----------------------|-----------|------------|----------------|
  | 536365    | 85123A   | WHITE HANGING HEART T-LIGHT HOLDER     | 6        | 2010-12-01 08:26:00  | 2.55      | 17850.0    | United Kingdom |
  | 536365    | 71053    | WHITE METAL LANTERN                    | 6        | 2010-12-01 08:26:00  | 3.39      | 17850.0    | United Kingdom |
  | 536365    | 84406B   | CREAM CUPID HEARTS COAT HANGER         | 8        | 2010-12-01 08:26:00  | 2.75      | 17850.0    | United Kingdom |
  
  *Sample snapshot of dataset*

- ### Data Cleaning and Preprocessing
  The purpose of this phase is to preprocess the data for clustering. Since there are no missing values in this dataset, the main goal is to identify 1) duplicate values and 2) remove canceled transactions from clustering analysis. The reason to remove canceled transactions is that since these transactions are not finished, they should be analyzed separately from other transactions in order to discover which items are most often canceled. 

- ### Exploratory Data Analysis
  After preprocessing, EDA is thus conducted in order to provide an overview of the current dataset. The following EDAs are performed:
  - #### Summary Statistics
    The following tables are the summary statistics of the qualitative and quantitative variables, respectively. Note that key and date columns (*InvoiceNo*, *StockCode*, *InvoiceDate*, *CustomerID*) are removed in this step of the analysis.
    | Attribute   | Count  | Unique | Top                                      | Freq  |
    |------------|--------|--------|------------------------------------------|-------|
    | Description | 401604 | 3896   | WHITE HANGING HEART T-LIGHT HOLDER      | 2058  |
    | Country    | 401604 | 37     | United Kingdom         | 356728 |
    
    *Summary statistics of qualitative variables*

    | Attribute   | Count    | Mean     | Std Dev     | Min       | 25%  | 50%  | 75%  | Max      |
    |------------|---------|---------|------------|----------|------|------|------|---------|
    | Quantity   | 401604.0 | 12.1833  | 250.2830   | -80995.0  | 2.00 | 5.00 | 12.00 | 80995.0  |
    | UnitPrice  | 401604.0 | 3.4741   | 69.7640    | 0.0       | 1.25 | 1.95 | 3.75  | 38970.0  |

    *Summary statistics of quantitative variables*

    Furthermore, we would also want to know the unique values corresponding to each feature:
    - Number of entries: 401604
    - Number of unique Customer IDs: 4372
    - Number of unique Countries: 37
    - Number of unique Products: 3684
    - Number of unique Transactions: 22190 \
      
    As a result, we can identify that the majority of the products of this retailer are low-priced items.

  - #### Country statistics
    In addition to basic summary statistics, we would also like to know where the customers and transactions have been made. Therefore, we can plot frequency histograms by transaction and customer, respectively. As a result, we can identify that the majority of our customers and transactions are based in the United Kingdom, with other Western European countries such as Germany and France following. 
    ![country](assets/country.png)
  
- ### Models
  Data analysis is performed throughout three phases: text vectorization, dimension reduction, and model fitting. 
  - #### Text Vectorization
    Prior to vectorizing, text is preprocessed with the _BertTokenizer_ function, which tokenizes chunks of text into individual tokens for subsequent vectorization. The following text vectorization is conducted with BERT, in particular, the _bert_base_uncased_ model. BERT was selected due to it being one of the state-of-the-art models in natural language processing. The output of BERT is a 768-dimension word embedding that represents the original text. GPU acceleration is also implemented to exploit parallel processing to reduce the runtime of the model. 
  - #### Dimension Reduction
    After vectorization, $X_i$ is a 768-dimension vector. As the complexity of some models scales exponentially with the dimension of $X_i$, dimension reduction is applied with Principal Component Analysis (PCA). PCA performs dimension reduction while maximizing the variance retained by the reduced dimensions. To select the number of principal components to be retained, a threshold of 90% is set, i.e. the principal components would capture 90% of the total variance of all features. After applying PCA, 286 principal components are retained, replacing the original word embeddings for subsequent model fitting.
    ![pca](assets/pca.png)
    *Cumulative explained variance graph of PCA. The red dashed line represents the 90% level of cumulative variance explained.*
  - #### Model Fitting
    - **Hyperparameter tuning:** Prior to fitting each model, hyperparameters related to each model are first optimized through Bayesian Optimization with the package _hyperopt_. The package _hyperopt_ is chosen due to its flexibility over its parameters in the optimization process.
    - **Model selection:**
      - *Logistic Regression with l2 penalty:* Logistic regression is first applied due to its computation efficiency. Regularization is applied to lower the variance of the model in trade of slightly increased bias. Between $elasticNet$, $l1$, and $l2$ penalties, the $l2$ penalty is selected due to 1) computational efficiency over $elasticNet$ in large datasets, and 2) features should not be sparse after PCA processing. The solver parameter was set to $lbfgs$ based on the characteristics elaborated in the [scikit documentation](https://scikit-learn.org/stable/modules/linear_model.html). The hyperparameter optimized in Logistic Regression is the parameter $C$, which represents the regularization strength. 
      - *Linear Support Vector Classification:* Linear SVC is a special instance of a Support Vector Machine (SVM). It is recommended to be implemented on large datasets in practice and also in [scikit's documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC). Linear SVC also applies a $l2$ regularization to the model by default. Therefore, the hyperparameter optimized in Linear SVC is also the parameter $C$, representing the regularization strength.
      - *Multilayer Perceptron:* Multilayer perceptron is a neural network that utilizes multiple hidden layers to capture the hierarchical features. Due to the high associated computational costs, hyperparameters were also selected based on best practices instead of extensive model tuning. A three-layer MLP is implemented with the $Adam$ optimizer, along with the loss function set as $sparseCategoricalCrossentropy$, a loss function often applied in datasets with integer $y$ labels.

    In addition, K-fold Cross-Validation with $K = 5$ is also implemented for model selection to lower the variance of the results.

- ### Results
  The results of the four models are summarized in the following table:

  | Model               | Accuracy   | Precision   | Recall    | F1-Score   |
  |:--------------------|:-----------|:------------|:----------|:-----------|
  | Logistic Regression | 0.740625   | 0.727775    | 0.740625  | 0.72769    |
  | LinearSVC           | 0.735234   | 0.723957    | 0.735234  | 0.713178   |
  | MLP                 | 0.744356   | 0.738371    | 0.744356  | 0.74052    |


- ### Limitations
  - The current category reduction process is based on domain knowledge with sampled data from each category. However, there may be some other set of categories that can further reduce inter-class similarity, therefore potentially further optimizing the performance of the classification models.
  - Hyperparameter tuning is limited due to constraints on computational resources. Implementing Bayesian Optimization on a more extensive range of hyperparameters may also further optimize the models.
  - Implementing single-label classification may lower the performance of our model, as some of the news entries have a more "ambiguous" class. For instance, President Trump's interactions with SNL hosts are often categorized as $COMEDY$ in the original labels. However, the keyword "Trump" may lead the model to misclassify that news entry into $General$ (which encompasses political news). Therefore, implementing multi-label classification may potentially address this issue by reflecting the nature of news having more than one category in real-world scenarios.  

- ### References
  [1] Simon Newman. "rtr1pc8i_0.jpg" Reuters Institute for the Study of Journalism, 31 October 2019, https://reutersinstitute.politics.ox.ac.uk/news/if-your-newsroom-not-diverse-you-will-get-news-wrong/ \
  [2] Rishabh Misra, News Category Dataset (Kaggle, 2022), https://www.kaggle.com/datasets/rmisra/news-category-dataset/data
