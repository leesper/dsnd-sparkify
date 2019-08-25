# dsnd-sparkify
My Solutions for Spark on machine learning

## Motivation

This is a machine learning project of data science nanodegree program. Sparkify is a music streaming startup, users can listen to the music, there are three types of users: guests, free tiers and premiums. Premium users need to pay for the service. They are the income source of the company. The app has recorded the user activities data in JSON format. Whenever a user log in, log out, listens to a music, upgrade, downgrade or do a cancellation confirmation, there will be a record generated. Our goal is to analyse the data and try to construct a classifer to predict whether a user will downgrade/do cancellation confirmation or not. So the company can offer discounts and incentives to make the business better.

## Requirements

1. Jupyter Notebook
2. A Spark running in local mode
3. pyspark package
4. pandas package
5. matplotlib package
6. seaborn package

If you want to train on much bigger data, you have to set up a spark cluster in AWS or do it your own.

## Files

1. Sparkify.ipynb: the notebook demonstrating the whole solution
2. report.pdf: the data science solution documenation
3. report.md: the original markdown file which generates the pdf above
4. LICENSE: MIT LICENSE file
5. README.md: this file
6. other jpeg and png fiiles: figures used in report