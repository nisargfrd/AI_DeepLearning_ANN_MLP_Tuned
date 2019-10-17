What is this repo about:
# AI_DeepLearning_ANN_MLP_Tuned:
This repo contains Jupyter Notebook for solving below mentioned problem statement using Perceptron and MLP models of ANN.</br>
I have also tuned the Hyperparameters: Learning rate and Batch-Size, for the best model.</br>

# Problem Statement:
Here we try to identify products at risk of backorder before the event occurs so that business has time to react.</br>
We'll use Artificial Neural Networks (ANN) - Multi-layer Perceptron (MLP) model for the same.</br>
We'll tune and find the best values for our Hyperparameters.

# What is a Backorder:
Backorders are products that are temporarily out of stock, but a customer is permitted to place an order against future inventory.</br>
A backorder generally indicates that customer demand for a product or service exceeds a company’s capacity to supply it.</br>
Back orders are both good and bad. Strong demand can drive back orders, but so can suboptimal planning.</br>

# Data:
Data file contains the historical data for the 8 weeks prior to the week we are trying to predict.</br>
The data was taken as weekly snapshots at the start of each week. Columns are defined as follows:

sku - Random ID for the product</br>
national_inv - Current inventory level for the part</br>
lead_time - Transit time for product (if available)</br>
in_transit_qty - Amount of product in transit from source</br>
forecast_3_month - Forecast sales for the next 3 months</br>
forecast_6_month - Forecast sales for the next 6 months</br>
forecast_9_month - Forecast sales for the next 9 months</br>
sales_1_month - Sales quantity for the prior 1 month time period</br>
sales_3_month - Sales quantity for the prior 3 month time period</br>
sales_6_month - Sales quantity for the prior 6 month time period</br>
sales_9_month - Sales quantity for the prior 9 month time period</br>
min_bank - Minimum recommend amount to stock</br>
potential_issue - Source issue for part identified</br>
pieces_past_due - Parts overdue from source</br>
perf_6_month_avg - Source performance for prior 6 month period</br>
perf_12_month_avg - Source performance for prior 12 month period</br>
local_bo_qty - Amount of stock orders overdue</br>
deck_risk - Part risk flag</br>
oe_constraint - Part risk flag</br>
ppap_risk - Part risk flag</br>
stop_auto_buy - Part risk flag</br>
rev_stop - Part risk flag</br>
went_on_backorder - Product actually went on backorder. This is the target value.</br>

# Right Error Metrics:
Recall/ TPR</br>
We'll use TPR.

# Prerequisites:
Keras, Tensorflow (CPU and GPU for better performance) needs to be installed on your machine to run the attached code/notebook.</br>
I've used "use_multiprocessing=True" option while fitting the model in Keras, for better performance.</br>
If your machine has low-end hardware, you can remove that option and then execute the code.</br>

# Happy Coding / AI Learning!
