# Customer Segmentation Business Analysis Report
https://archive.ics.uci.edu/dataset/352/online+retail

- ### Project Objective and Overview
    This project aims to compile a business analysis report through segmentation analysis of an online retail dataset. This project first deploys classical EDA methods to provide an overview of the dataset. Subsequently, RFM (Recency, Frequency, Monetary) indexes are created, which perform as the features used for segmentation. In addition, cancelation rate and customer activity indexes are also created to provide additional information on consumer characteristics. A  business analysis report aimed at providing next-step suggestions is derived from the results of segmentation analysis.

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
    - Number of unique Transactions: 22190 
      
    As a result, we can identify that the majority of the products of this retailer are low-priced items.

  - #### Country Statistics
    In addition to basic summary statistics, we would also like to know where the customers and transactions have been made. Therefore, we can plot frequency histograms by transaction and customer, respectively. As a result, we can identify that the majority of our customers and transactions are based in the United Kingdom, with other Western European countries such as Germany and France following. 
    ![country](assets/country.png)
    *Frequency histogram by transaction (left) and by customer (right)*

  - #### Product Statistics
    To analyze the top products sold, we can identify the top products sold by quantity and by total value. We can identify that the top products sold either by quantity or by value are similar, such as *WHITE HANGING HEART T_LIGHT HOLDER*, *REGENCY CAKESTAND 3 TIER*, and *JUMBO BAG RED RETROSPOT* are all in the top 5 products by quantity and by value.
    ![product](assets/product.png)

  - #### Canceled Product Statistics
    In addition to successful transactions, identifying products that are frequently canceled could also provide insights into lowering cancellation rates in the future. Similarly, we can also analyze most canceled products by both quantity and value. We can observe that the top product by value *PAPER CRAFT, LITTLE BIRDIE* is also the top canceled product by value. Further analysis into reasons why customers return these products could potentially improve both customer experience and reduce costs derived from transaction cancellations. 
    ![canceled](assets/canceled.png)

- ### Index Creation
  - #### RFM Indexes
    RFM indexes are commonly used in customer segmentation, serving as a basis of segmentation based on consumer purchasing behavior. In particular, the indexes are operationalized with the following methods:

    - **Recency Index:**
    The Recency Index represents how recently a customer has completed a purchase. In this analysis, it is operationalized by computing the difference in days between the current date and the most recent purchase. However, due to the limitation of the timeframe of this dataset, the current date is set to be the date of the most recent transaction that occurred in the dataset.

        $$R_i = (\text{LatestDate} - \text{LastPurchaseDate}_{i}), \quad \forall i \in \text{Customers}$$
    
    - **Frequency Index:**
      The Frequency Index represents how frequently a customer has been purchasing. In this analysis, it is operationalized by counting the number of unique invoice numbers associated with a particular customer. The reason that the number of unique invoice numbers is used is that even if an invoice number is related to multiple products, each invoice number only represents one transaction. Therefore, to accurately reflect the purchasing frequency of a customer, unique invoice numbers should be counted since they represent unique transactions. 

        $$F_i = \sum_{j} x_{ij}, \quad \forall i \in \text{Customers}, \forall j \in \text{Transactions}$$

      In which $X_{ij}$ denotes a binary variable that is equal to one if that transaction's invoice number is unique.

    - **Monetary Index:**
      The Monetary Index represents the total monetary value spent by a customer. In this analysis, it is operationalized by summing the total revenue generated from all transactions for a specific customer. The variable $Monetary_i$ denotes the total value of a particular customer.

      $$M_i = \sum_{j} \text{Monetary}_{i}, \quad \forall i \in \text{Customers}$$

    - **Results:** After creating RFM Indexes, we can observe the distributions of the RFM values using histograms. However, all three indexes are heavily skewed as a result. Prior to log transformation. we first removed customers with a negative monetary value (n=42). The reason for removing such customers is that, from the retailer's perspective, these customers generate negative value for the retailer. Therefore, if these customers are placed in the same pool as other customers, the clustering algorithm may group these customers into the same cluster. In response, by removing these customers prior to clustering, we could reduce the noise created by these customers and focus on segmenting customers who generate revenue for the retailer. 
      ![rfm_raw](assets/rfm_raw.png)
      *Histograms of raw RFM Indexes*
      
      In response, we can perform log transformation on all three indexes to reduce the skewness of their distributions. The following graph is the results after performing log transformation:
      ![rfm_transformed](assets/rfm_transformed.png)
      *Histograms of log transformed RFM Indexes* 

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
