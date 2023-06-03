from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler

def compute_x(sample_df, row_to_analize, all_features, important_features, main_feature):
    """
    Create the columns needed for variables x+j, x-j
    Given a subset of important_features, those columns will take the value of the sample (the row for the dataset currently working on)
    While the other columns (and the rest of the columns in all_features) will take the value from row_to_analize.
    The only difference between x+j and x-j is that for x+j the main_feature column will take the value from row_to_analize while x-j will take the value from the sample
    
    :param sample_df: dataframe were we shall sample examples
    :param row_to_analize: example row that we wish to analize the impact of each important feature. It should be a dictionary with a key for each value in all_features
    :param all_features: all features needed to compute the prediction
    :param important_features: all features that we wish to compute the shapley values and we can swap values between row and sample
    :param main_feature: feature that we wish to compute the shapley value
    :return: 
    """
    
    #Check which columns we will pick out from the sample
    #first randomly sort the features, then check how many to choose
    df = sample_df.withColumn('features_permutations', F.shuffle(F.array(*[F.lit(col) for col in important_features if col != main_feature])))\
           .withColumn('n_features', F.floor(F.rand(seed=42) * (len(important_features)+1)))\
           .withColumn('features_pick', F.expr("slice(features_permutations, 1, n_features)"))
    
    #compute columns for x+-j from important_features2, these will be called col+"_select"
    for col in important_features:
        df = df.withColumn(col + "_select", F.when(F.array_contains(F.col('features_pick'), col), F.col(col)).otherwise(row_to_analize[col]))
        
    #create the column for x+-j with respect to main_feature
    df = df.withColumn(main_feature + "_plus", F.lit(row_to_analize[main_feature])) #plus will take the row value
    df = df.withColumn(main_feature + "_minus", F.col(main_feature)) #minus will take the sample value
    
    #all other variables outside of important_features will take the value from row_to_analize
    non_imp_vars = [col for col in all_features if col not in important_features]
    for col in non_imp_vars:
        df = df.withColumn(col + "_select", F.lit(row_to_analize[col]))
        
    #clean output
    df = df.select([main_feature + "_plus", main_feature + "_minus"] + [col + "_select" for col in all_features if col != main_feature])    
        
    return df


def predict(df, all_features, main_feature, prediction_fun, features_col='features', prediction_col='prediction'):
    """
    Build x+j, x-j by creating the appropriate VectorAssembler.
    x+-j will be formed by all columns from all_features ending in "_select" and the column ending in "_plus" or "_minus" (respectively)
    Then predict both
 
    :param df: dataframe with all features needed to create x+j, x-j
    :param all_features: all features needed to compute the prediction
    :param main_feature: feature that we wish to compute the shapley value
    :param features_col: column that stores the vector that is input for prediction_fun 
    :param prediction_col: column with the result of prediction_fun
    :return:
    """
    pred_df = df
    for typ in ['_plus','_minus']:
          #build feature vector, the only thing that changes is in "main_feature" we have a different value
          input_cols = [col + typ if col == main_feature else col + "_select" for col in all_features]
          vector_assembler = VectorAssembler(inputCols=input_cols, outputCol=features_col)

          #apply model to x+-j
          pred_df = vector_assembler.transform(pred_df)
          pred_df = prediction_fun(pred_df)
          pred_df = pred_df.withColumn(prediction_col + typ, F.col(prediction_col))
          pred_df = pred_df.drop(features_col,prediction_col)
        
    return pred_df
  
def compute_marginal_contribution(df, prediction_col):
    """
    Compute the marginal contribution of each sample and then take the average. And this will be the shapley value
    
    :param df: dataframe where each row is the prediction of x+j and x-j
    :param prediction_col: column with the result of prediction_fun   
    """
    #marginal contribution will be f(x+j) - f(x-j)
    df = df.withColumn('marginal_contribution', F.col(prediction_col+'_plus') - F.col(prediction_col+'_minus'))
    
    #want average
    return df.select(F.avg('marginal_contribution')).toPandas().iloc[0][0]


def compute_shap_values(input_df, row_to_analize, all_features, important_features, prediction_fun, features_col='features', prediction_col='prediction'):
    """
    Compute shapley value for each feature in important_features
    
    :param input_df: dataset were we shall sample examples
    :param row_to_analize: example row that we wish to analize the impact of each important feature. It should be a dictionary with a key for each value in all_features
    :param all_features: all features needed to compute the prediction
    :param important_features: all features that we wish to compute the shapley values
    :param prediction_fun: how to compute the prediction of an example
    :param features_col: column that stores the vector that is input for prediction_fun 
    :param prediction_col: column with the result of prediction_fun
    :return: dictionary with (feature, shapley value)
    """
    shap_values = {}
    
    for main_feature in important_features:
        df = compute_x(input_df, row_to_analize, all_features, important_features, main_feature)
        
        df.persist().count() #persist since we create features twice
        df = predict(df, all_features, main_feature, prediction_fun, features_col, prediction_col)
        res = compute_marginal_contribution(df, prediction_col)

        #store in dictionary
        shap_values[main_feature] = res
        print("{}:{}".format(main_feature, res)) # print value just in case this takes too long
        df.unpersist()
        
    return shap_values
