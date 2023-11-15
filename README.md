# Project Name: House Price Prediction - Advanced regression Case Study 

## Table of Contents

- [General Info](#general-information)
- [Technologies Used](#technologies-used)
- [Conclusions](#conclusions)
- [Acknowledgements](#acknowledgements)


## General Information

### Project Background

> A US-based housing company named Surprise Housing has decided to enter the Australian market. The company uses data analytics to purchase houses at a price below their actual values and flip them on at a higher price. The company is looking at prospective properties to buy to enter the market. 

### The business is curious about:

The company wants to know:

    Which variables are significant in predicting the price of a house, and

    How well those variables describe the price of a house.

Also, determine the optimal value of lambda for ridge and lasso regression.

### Project Statement

> This research aims to build a regression model using regularisation in order to predict the actual value of the prospective properties and decide whether to invest in them or not.



## Conclusions

> 	The below mentioned variables are significant in predicting the price

1.	GrLivArea
2.	OverallQual
3.	YearBuilt
4.	TotalBsmtSF
5.	OverallCond
6.	LotArea
7.	BsmtFinSF1
8.	RoofMatl_CompShg
9.	SaleType_New
10.	Exterior2nd_CmentBd
 
> Answer-

              Ridge Regression  Lasso Regression

R2 score(Train) 8.79             8.76

R2 score(Test) 8.31              8.32

Metric	Linear Regression	Ridge Regression	Lasso Regression	
0	R2 Score (Train)	8.98E-01	8.80E-01	8.76E-01
1	R2 Score (Test)	7.81E-01	8.31E-01	8.33E-01
2	RSS (Train)	5.41E+11	6.37E+11	6.56E+11
3	RSS (Test)	5.17E+11	3.98E+11	3.94E+11
4	MSE (Train)	2.30E+04	2.50E+04	2.53E+04
5	MSE (Test)	3.43E+04	3.01E+04	3.00E+04

 

## Technologies Used

- Pandas 
- Seaborn 
- MatplotLib 
- Missingno
- scikit-learn
- statsmodel

## Acknowledgements

This project was done as part of Case study for UpGrad IITB Programme on Artifical Inteligince and Machine Learning.
