What is this repo about:
# AI_DeepLearning_ANN_MLP_Tuned:
This repo contains Jupyter Notebook for solving below mentioned problem statement using Perceptron and MLP models of ANN.
I have also tuned the Hyperparameters: Learning rate and Batch-Size, for the best model.

# Problem Statement:
Here we try to identify products at risk of backorder before the event occurs so that business has time to react.
We'll use Artificial Neural Networks (ANN) - Multi-layer Perceptron (MLP) model for the same.
We'll tune and find the best values for our Hyperparameters.

# What is a Backorder:
Backorders are products that are temporarily out of stock, but a customer is permitted to place an order against future inventory.
A backorder generally indicates that customer demand for a product or service exceeds a company’s capacity to supply it.
Back orders are both good and bad. Strong demand can drive back orders, but so can suboptimal planning.

# Data:
Data file contains the historical data for the 8 weeks prior to the week we are trying to predict.
The data was taken as weekly snapshots at the start of each week. Columns are defined as follows:

sku - Random ID for the product
national_inv - Current inventory level for the part
lead_time - Transit time for product (if available)
in_transit_qty - Amount of product in transit from source
forecast_3_month - Forecast sales for the next 3 months
forecast_6_month - Forecast sales for the next 6 months
forecast_9_month - Forecast sales for the next 9 months
sales_1_month - Sales quantity for the prior 1 month time period
sales_3_month - Sales quantity for the prior 3 month time period
sales_6_month - Sales quantity for the prior 6 month time period
sales_9_month - Sales quantity for the prior 9 month time period
min_bank - Minimum recommend amount to stock
potential_issue - Source issue for part identified
pieces_past_due - Parts overdue from source
perf_6_month_avg - Source performance for prior 6 month period
perf_12_month_avg - Source performance for prior 12 month period
local_bo_qty - Amount of stock orders overdue
deck_risk - Part risk flag
oe_constraint - Part risk flag
ppap_risk - Part risk flag
stop_auto_buy - Part risk flag
rev_stop - Part risk flag
went_on_backorder - Product actually went on backorder. This is the target value.

# Right Error Metrics:
Recall/ TPR
We'll use TPR.

# Prerequisites:
Keras, Tensorflow (CPU and GPU for better performance) needs to be installed on your machine to run the attached code/notebook.
I've used "use_multiprocessing=True" option while fitting the model in Keras, for better performance.
If your machine has low-end hardware, you can remove that option and then execute the code.

# Happy Coding / AI Learning!
