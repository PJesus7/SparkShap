# SparkShap
Another implementation of computing Shapley values in spark.

Shapley values are explained [here](https://christophm.github.io/interpretable-ml-book/shapley.html#estimating-the-shapley-value)

# Motivation

The recommended [SHAP repository](https://github.com/slundberg/shap) works mostly with scikit-learn models and with some Spark models. But since I have categorical variables, the solution proposed by this [article](https://towardsdatascience.com/parallelize-your-massive-shap-computations-with-mllib-and-pyspark-b00accc8667c) didn't work.

Other packages/repositories I saw using SHAP with spark had some sort of udf/local component, i decided to implement this myself. This is based on the [following article](https://tryexceptfinally.hashnode.dev/machine-learning-interpretability-shapley-values-with-pyspark-16ffd87227e3), where i noticed that the udf could be rewritten using only native pyspark code.

So notice that the only function implemented is how to compute the Shapley values for an example.

# How to use

Do note that this is a very crude code, but I just wanted to share with others that were having the same trouble as I with computing shapley values using a pyspark model.

- This code is implemented using pyspark 2.4.3.
- It assumes that you have a trained pyspark model
- It assumes that in your training pipeline you have a **VectorAssembler** immediately before you call your model
- You should know the name of the features of the **VectorAssembler** so that you can compute the SHAP values on those

The notebook has an example dataset with a trained model and on how to use the functions.