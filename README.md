# BlackFridayDataHackathon
Analytics Vidhya's Black Friday Data Hackathon

# Problem Statement

The challenge was to predict purchase prices of various products purchased by customers based on historical purchase patterns. The data contained features like age, gender, marital status, categories of products purchased, city demographics etc.
http://datahack.analyticsvidhya.com/contest/black-friday-data-hack

# My approach for the hackathon is as follows:

1. Looked into levels of data and converted all variables into factors. 
2. Imputed missing values with '999' and converted such variables into factors
3. Ran basic Multi Linear regression and Submitted benchmark model predictions
4. Ran a basic random forest, Xgboost, GLM, GBM, Deep Learning algorithms by excluding USER_ID and PRODUCT_ID.
5. Got a RMSE of 2888, Public Leader Board Ranking 66/162.

# Learnings
1. Using H2O packages
2. Feature engineering needs to be improved
3. Reviewed code of top rankers. Hoping these learnings will improve my next LB ranking to top 10% :)
