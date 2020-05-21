from collections import Counter
import logging
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef, r2_score, f1_score

from mindsdb.libs.constants.mindsdb import *
from mindsdb.libs.helpers.text_helpers import clean_str_data_to_float_or_date


def compute_value_distribution_score(stats, columns, col_name):
    """
    # Looks at the histogram and transforms it into a proability mapping for each
    bucket, then generates a quality score (value_distribution_score) based on that

    :param stats: The stats extracted up until this point for all columns
    :param columns: All the columns
    :param col_name: The name of the column we should compute the new stats for
    :return: Dictioanry containing:
        bucket_probabilities: A value distribution score, ranges from 1 to 0, where 1 is lowest quality and 0 is highest quality.
        value_distribution_score: A dictioanry of with the probabilities that a value values in a bucket, for each of he buckets in the histogram
    """

    bucket_probabilities = {}
    pair = stats[col_name]['histogram']
    total_vals = sum(pair['y'])
    for i in range(len(pair['x'])):
        bucket_probabilities[pair['x'][i]] = pair['y'][i]/total_vals

    probabilities = list(bucket_probabilities.values())

    max_probability = max(probabilities)
    max_probability_key = max(bucket_probabilities, key=lambda k: bucket_probabilities[k])

    value_distribution_score = 1 - np.mean(probabilities)/max_probability

    data = {
        'bucket_probabilities': bucket_probabilities
        ,'value_distribution_score': round(10 * (1 - value_distribution_score))
        ,'max_probability_key': max_probability_key
        ,'value_distribution_score_description': """
        This score can indicate either biasing towards one specific value in the column or a large number of outliers. So it is a reliable quality indicator but we can't know for which of the two reasons.
        """
    }

    return data


def compute_duplicates_score(stats, columns, col_name):
    """
    # Looks at the set of distinct values for all the data and computes a quality
    socre based on how many of the values are duplicates

    :param stats: The stats extracted up until this point for all columns
    :param columns: All the columns
    :param col_name: The name of the column we should compute the new stats for
    :return: Dictioanry containing:
        nr_duplicates: the nr of cells which contain values that are found more than once
        duplicates_percentage: % of the values that are found more than once
        duplicates_score: a quality based on the duplicate percentage, ranges from 1 to 0, where 1 is lowest quality and 0 is highest quality.
    """

    occurances = Counter(columns[col_name])
    values_that_occur_twice_or_more = filter(lambda val: occurances[val] < 2, occurances)
    nr_of_occurances = map(lambda val: occurances[val], values_that_occur_twice_or_more)
    nr_duplicates = sum(nr_of_occurances)
    data = {
        'nr_duplicates': nr_duplicates
        ,'duplicates_percentage': nr_duplicates*100/len(columns[col_name])
        ,'duplicates_score_description':"""
        The duplicates score consists in the % of duplicate values / 100. So, it can range from 0 (no duplicates) to 1 (all the values have one or more duplicates). This score being large, on it's own, is not necessarily an indicator that your data is of poor quality.
        """
    }

    if stats[col_name]['data_type'] != DATA_TYPES.CATEGORICAL and stats[col_name]['data_type'] != DATA_TYPES.DATE:
        data['duplicates_score'] = data['duplicates_percentage']/100
    else:
        data['c'] = 0

    return data

def compute_empty_cells_score(stats, columns, col_name):
    """
    # Creates a quality socre based on the percentage of empty cells (empty_percentage)

    :param stats: The stats extracted up until this point for all columns
    :param columns: All the columns
    :param col_name: The name of the column we should compute the new stats for
    :return: Dictioanry containing:
        empty_cells_score: A quality score based on the nr of empty cells, ranges from 1 to 0, where 1 is lowest quality and 0 is highest quality.
    """

    return {'empty_cells_score': round(10 * (1 - stats[col_name]['empty_percentage']/100))
            ,'empty_cells_score_description':"""This score is computed as the % of empty values / 100. Empty values in a column are always bad for training correctly on that data."""}

def compute_data_type_dist_score(stats, columns, col_name):
    """
    # Creates a quality socre based on the data type distribution, this score is based on
    the difference between the nr of values with the "main" data type
    and all the nr of values with all other data types

    :param stats: The stats extracted up until this point for all columns
    :param columns: All the columns
    :param col_name: The name of the column we should compute the new stats for
    :return: Dictioanry containing:
        data_type_distribution_score: A quality score based on the nr of empty cells, ranges from 1 to 0, where 1 is lowest quality and 0 is highest quality.
    """

    vals = stats[col_name]['data_type_dist'].values()
    principal = max(vals)
    total = len(columns[col_name])
    data_type_dist_score = (total - principal)/total
    return {'data_type_distribution_score': round(10 * (1 - data_type_dist_score))
    ,'data_type_distribution_score_description':"""
    This score indicates the amount of data that are not of the same data type as the most commonly detected data type in this column. Note, the most commonly occuring data type is not necessarily the type mindsdb will use to label the column when learning or predicting.
    """}


def compute_z_score(stats, columns, col_name):
    """
    # Computes the z_score for each value in our column.
    # This score represents the distance from the mean over the standard deviation.
    # Based on this, compute a quality metrics.

    :param stats: The stats extracted up until this point for all columns
    :param columns: All the columns
    :param col_name: The name of the column we should compute the new stats for
    :return: Dictioanry containing:
        z_score_outliers: The indexs of values which we consider outliers based on the z score
        mean_z_score: The mean z score for the column
        z_test_based_outlier_score: A quality score based on the nr of outliers as determined by their z score, ranges from 1 to 0, where 1 is lowest quality and 0 is highest quality.
    """
    if stats[col_name]['data_type'] != DATA_TYPES.NUMERIC:
        return {}

    col_data = columns[col_name]

    z_scores = list(map(abs,(st.zscore(col_data))))
    threshold = 3
    z_score_outlier_indexes = [i for i in range(len(z_scores)) if z_scores[i] > threshold]
    data = {
        'z_score_outliers': z_score_outlier_indexes
        ,'mean_z_score': round(10 * (1 - np.mean(z_scores)))
        ,'z_test_based_outlier_score': round(10 * (1 - len(z_score_outlier_indexes)/len(col_data)))
        ,'z_test_based_outlier_score_description':"""
        This score indicates the amount of data that are 3 STDs or more away from the mean. That is to say, the amount of data that we consider to be an outlir. A hgih z socre means your data contains a large amount of outliers.
        """
    }
    return data

def compute_lof_score(stats, columns, col_name):
    """
    # Uses LocalOutlierFactor (a KNN clustering based method from sklearn)
    to determine outliers within our column
    # All data that has a small score after we call `fit_predict` has a high chance of being an outlier
    based on the distance from the clusters created by LOF

    :param stats: The stats extracted up until this point for all columns
    :param columns: All the columns
    :param col_name: The name of the column we should compute the new stats for
    :return: Dictioanry containing:
        lof_outliers: The indexs of values which we consider outliers based on LOF
        lof_based_outlier_score: A quality score based on the nr of outliers as determined by their LOF score, ranges from 1 to 0, where 1 is lowest quality and 0 is highest quality.
    """

    if stats[col_name]['data_type'] != DATA_TYPES.NUMERIC:
        return {}

    np_col_data = np.array(columns[col_name]).reshape(-1, 1)
    lof = LocalOutlierFactor(contamination='auto')
    outlier_scores = lof.fit_predict(np_col_data)
    outlier_indexes = [i for i in range(len(columns[col_name])) if outlier_scores[i] < -0.8]

    return {
        'lof_outliers': outlier_indexes
        ,'lof_based_outlier_score': round(10 * (1 - len(outlier_indexes)/len(columns[col_name])))
        ,'percentage_of_log_based_outliers': (len(outlier_indexes)/len(columns[col_name])) * 100
        ,'lof_based_outlier_score_description':"""
        The higher this score, the more outliers your dataset has. This is based on distance from the center of 20 clusters as constructed via KNN.
        """
    }


def compute_similariy_score(stats, columns, col_name):
    """
    # Uses equality between values in the same position to determine up what % of their cells two columns are identical

    :param stats: The stats extracted up until this point for all columns
    :param columns: All the columns
    :param col_name: The name of the column we should compute the new stats for
    :return: Dictioanry containing:
        similarities: How similar this column is to other columns (with 0 being completely different and 1 being an exact copy).
        similarity_score: A score equal to the highest similarity found, ranges from 1 to 0, where 1 is lowest quality and 0 is highest quality.
    """
    col_data = columns[col_name]

    similarities = []
    for other_col_name in columns.columns:
        if other_col_name == col_name:
            continue
        else:
            # @TODO Figure out why computing matthews_corrcoef is so slow, possibly find a better implementation and replace it with that. Matthews corrcoef code was: similarity = matthews_corrcoef(list(map(str,col_data)), list(map(str,columns[other_col_name])))
            similarity = 0
            X1 = list(map(str, col_data))
            X2 = list(map(str, columns[other_col_name]))
            for ii in range(len(X1)):
                if X1[ii] == X2[ii]:
                    similarity += 1

            similarity = similarity/len(X1)
            similarities.append((other_col_name,similarity))


    max_similarity = max(map(lambda x: x[1], similarities))
    most_similar_column_name = list(filter(lambda x: x[1] == max_similarity, similarities))[0][0]

    if max_similarity < 0:
        max_similarity = 0

    return {
        'max_similarity': max_similarity,
        'similarities': similarities,
        'similarity_score': round(10 * (1 - max_similarity)),
        'most_similar_column_name': most_similar_column_name,
        'similarity_score_description': """
        This score is simple element-wise equality applied between this column and all other column.
        The score * 100 is the number of values which are similar in the column that is most similar to the scored column.
        """
    }


def compute_clf_based_correlation_score(stats, columns, col_name):
    """
    # Tries to find correlated columns by trying to predict the values in one
    column based on all the others using a simple DT classifier
    # The purpose of this is not to see if a column is predictable, but rather, to
    see if it can be predicted accurately based on a single other column, or if
    all other columns play an equally important role
    # A good prediction score, based on all the columns, doesn't necessarily imply
    a correlation between columns, it could just mean the data is very garnular (few repeat values)

    :param stats: The stats extracted up until this point for all columns
    :param columns: All the columns
    :param col_name: The name of the column we should compute the new stats for
    :return: Dictioanry containing:
        correlation_score: A score equal to the prediction accuracy * the importance (from 0 to 1)
            of the most imporant column in making said prediciton,
            ranges from 1 to 0, where 1 is lowest quality and 0 is highest quality.
        highest_correlation: The importance of the most_correlated_column in the DT classifier
        most_correlated_column: The column with which our column is correlated most based on the DT
    """
    full_col_data = columns[col_name]

    dt_clf = DecisionTreeClassifier()

    other_feature_names = []
    other_features = []
    for other_col_name in columns.columns:
        if other_col_name == col_name:
            continue

        other_feature_names.append(other_col_name)
        le = LabelEncoder()
        _stringified_col = list(map(str, columns[other_col_name]))
        le.fit(_stringified_col)
        other_features.append(list(le.transform(_stringified_col)))

    other_features_t = np.array(other_features, dtype=object).transpose()

    le = LabelEncoder()
    _stringified_col = list(map(str, full_col_data))
    le.fit(_stringified_col)
    y = le.transform(_stringified_col)
    dt_clf.fit(other_features_t, y)
    prediction_score = dt_clf.score(other_features_t, y)
    corr_scores = list(dt_clf.feature_importances_)
    highest_correlated_column = max(corr_scores)
    return {
        'correlation_score': round(10 * (1 - prediction_score * highest_correlated_column))
        ,'highest_correlation': max(corr_scores)
        ,'most_correlated_column': other_feature_names[corr_scores.index(max(corr_scores))]
        ,'similarity_score_description': """
        A high value for this score means that two of your columns are highly similar. This is done by trying to predict one column using the other via a simple DT.
        """
    }

def compute_consistency_score(stats, col_name):
    """
    # Attempts to determine the consistency of the data in a column
    by taking into account the ty[e distribution, nr of empty cells and duplicates

    :param stats: The stats extracted up until this point for all columns
    :param col_name: The name of the column we should compute the new stats for
    :return: Dictioanry containing:
        consistency_score: The socre, ranges from 1 to 0, where 1 is lowest quality and 0 is highest quality
    """
    col_stats = stats[col_name]
    if 'duplicates_score' in col_stats:
        consistency_score = (col_stats['data_type_distribution_score'] + col_stats['empty_cells_score'])/2.5 + col_stats['duplicates_score']/5
    else:
        consistency_score = (col_stats['data_type_distribution_score'] + col_stats['empty_cells_score'])/2
    return {'consistency_score': consistency_score
        , 'consistency_score_description': """
    A high value for this score indicates that the data in a column is not very consistent, it's either missing a lot of valus or the type of values it has varries quite a lot (e.g. combination of strings, dates, integers and floats).
    The data consistency score is mainly based upon the Data Type Distribution Score and the Empty Cells Score, the Duplicates Score is also taken into account if present but with a smaller (2x smaller) bias.
    """}

def compute_redundancy_score(stats, col_name):
    """
    # Attempts to determine the redundancy of the column by taking into account correlation and
    similarity with other columns

    :param stats: The stats extracted up until this point for all columns
    :param col_name: The name of the column we should compute the new stats for
    :return: Dictioanry containing:
        consistency_score: The socre, ranges from 1 to 0, where 1 is lowest quality and 0 is highest quality
    """
    col_stats = stats[col_name]
    redundancy_score = (col_stats['similarity_score'])/1
    return {'redundancy_score': redundancy_score
        ,'redundancy_score_description':"""
        A high value in this score indicates the data in this column is highly redundant for making any sort of prediction, you should make sure that values heavily related to this column are not already expressed in another column (e.g. if this column is a timestamp, make sure you don't have another column representing the exact same time in ISO datetime format).
        The value is based in equal part on the Similarity Score and the Correlation Score.
        """}

def compute_variability_score(stats, col_name):
    """
    # Attempts to determine the variability/randomness of a column by taking into account
    the z and lof outlier scores and the value distribution score (histogram biasing towards a few buckets)

    :param stats: The stats extracted up until this point for all columns
    :param col_name: The name of the column we should compute the new stats for
    :return: Dictioanry containing:
        consistency_score: The socre, ranges from 1 to 0, where 1 is lowest quality and 0 is highest quality
    """
    col_stats = stats[col_name]
    if 'lof_based_outlier_score' in col_stats and 'z_test_based_outlier_score' in col_stats:
        variability_score = (col_stats['z_test_based_outlier_score'] + col_stats['lof_based_outlier_score']
         + col_stats['value_distribution_score'])/3
    else:
        variability_score = col_stats['value_distribution_score']/2

    return {'variability_score': variability_score
    ,'variability_score_description':"""
    A high value for this score indicates the data in this column seems to be very variable, indicating a large possibility of some random noise affecting your data. This could mean that the values for this column are not collected or processed correctly.
    The value is based in equal part on the Z Test based outliers score, the LOG based outlier score and the Value Distribution Score.
    """}


def compute_data_quality_score(stats, col_name):
    """
    # Attempts to determine the quality of the column through aggregating all quality score
    we could compute about it

    :param stats: The stats extracted up until this point for all columns
    :param col_name: The name of the column we should compute the new stats for
    :return: Dictioanry containing:
        quality_score: An aggreagted quality socre that attempts to asses the overall quality of the column,
            , ranges from 1 to 0, where 1 is lowest quality and 0 is highest quality.
        bad_scores: The socres which lead to use rating this column poorly
    """

    col_stats = stats[col_name]
    scores = ['consistency_score', 'redundancy_score', 'variability_score']
    quality_score = 0
    for score in scores:
        quality_score += col_stats[score]
    quality_score = quality_score/len(scores)

    return {'quality_score': quality_score
    ,'quality_score_description':"""
    The higher this score is, the lower the quality of a given column.
    """}


def compute_predictive_power_score(stats, columns, col_name,
                                   combination_threshold=0.3):
    """
        # Attempts to estimate if the column `col_name` can be predicted using
        other columns, by fitting SGDRegressor and SGDClassifier on each valid pair of columns.
        If the column can be predicted from other columns, it likely carries
        no useful information.

        :param stats: The stats extracted up until this point for all columns
        :param columns: All the columns
        :param col_name: The name of the column we should compute the new stats for
        :param combination_threshold: column names, for which obtained score is
            greater than this value will be included in `predictive_combination` (see return).
        :return: Dictioanry containing:
            max_predictive_power: maximum score obtained by trying to predict this column from other columns.
                r2_score in case the column is numeric, f1_score in case the column is categorical.
            max_predictive_power_col: the column, for which maximum score was obtained.
                Can be None, if no column with score >= 0 was found.
            predictive_power_score: user-friendly score on a range from 0 to 10,
                where 10 indicates that the column can't be inferred from other columns,
                0 indicates that the column is easy to obtain from other columns (max_predictive_power ~ 1.0).
            predictive_combination: list of columns, for which obtained scores are
                above `combination_threshold`.
            predictive_power_score_description: description.
    """

    def validate_y(y_col, y_dtype):
        if sum(pd.isnull(y_col)) / len(y_col) >= 0.3:
            logging.warning('Column %s contains at least 30\% nan values, '
                            'predictive power score might not be reliable',
                            col_name)

        if np.unique(y_col[~pd.isnull(y_col)]).shape[0] == 1:
            return {
                'predictive_power_score_description': f'Column contains only a single value, '
                f'predictive power score not supported'
            }

        if y_dtype not in (DATA_TYPES.NUMERIC,
                           DATA_TYPES.CATEGORICAL,
                           DATA_TYPES.DATE):
            return {
                'predictive_power_score_description': f'Predictive power score for '
                f'data type {y_dtype} not supported'
            }

        if y_dtype == DATA_TYPES.DATE:
            y_subtype = stats[col_name]['data_subtype']
            if y_subtype == DATA_SUBTYPES.DATE:
                return {
                    'predictive_power_score_description': f'Predictive power score for '
                    f'data type {DATA_TYPES.DATE} not supported'
                }

    def preprocess_y(y_col, y_dtype):
        # Only evaluate x and y pairs where both are non-null
        non_null_index = ~pd.isnull(y_col)
        y_col = y_col[non_null_index]

        if y_dtype == DATA_TYPES.CATEGORICAL:
            y_encoder = LabelEncoder()
            y_col = y_encoder.fit_transform(y_col)
            return y_col, non_null_index
        elif y_dtype == DATA_TYPES.DATE:
            # Attempt to convert timestamps to int
            y_col = np.array([pd.to_datetime(y, infer_datetime_format=True).value for y in y_col])

        y_col = StandardScaler().fit_transform(y_col.reshape(-1, 1)).flatten()
        return y_col, non_null_index

    def is_valid_x(x_col, x_dtype, x_subtype):
        if np.unique(x_col[~pd.isnull(x_col)]).shape[0] == 1:
            # Contains only a single value
            return False

        if ((x_dtype not in (DATA_TYPES.NUMERIC,
                             DATA_TYPES.CATEGORICAL,
                             DATA_TYPES.DATE))
            or (x_dtype in DATA_TYPES.DATE and x_subtype == DATA_SUBTYPES.DATE)):
            # Cant compute predictive power score
            return False
        return True

    def preprocess_x(x_col, x_dtype):
        if x_dtype == DATA_TYPES.CATEGORICAL:
            x_col = x_col.astype(str)
            x_col[pd.isnull(x_col)] = 'NaN'
            x_encoder = LabelEncoder()
            x_col = x_encoder.fit_transform(x_col).reshape(-1, 1)
            return x_col

        if x_dtype == DATA_TYPES.DATE:
            # Attempt to convert timestamps to int
            x_col = np.array([pd.to_datetime(x, infer_datetime_format=True).value
                              for x in x_col.flatten()])\
                .reshape(-1, 1)

        x_col = SimpleImputer().fit_transform(x_col)
        x_col = StandardScaler().fit_transform(x_col)
        return x_col

    def obtain_score(x_col, y_col, model_cls, scoring_function):
        x_train, x_test, y_train, y_test = train_test_split(x_col, y_col)
        model = model_cls()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        score = scoring_function(y_test, y_pred)
        # R^2 can be negative, but we are only interested if its >= 0
        score = max(score, 0)
        return score

    def to_user_friendly_score(score, metric='r2_score'):
        if metric == 'r2_score':
            score = max(score, 0)

        # Map linearly:
        # source score of 1 to 0 user friendly score
        # 0 to 10
        score = -1 * score
        user_friendly_score = int(round(((score - (-1)) * (10 - 0)) / (0 - (-1))))
        return user_friendly_score

    max_predictive_power = 0
    max_predictive_power_col = None

    y_dtype = stats[col_name]['data_type']
    y_col = columns[col_name].values

    result = validate_y(y_col, y_dtype)
    if result:
        return result

    y_col, non_null_index = preprocess_y(y_col, y_dtype)

    model_cls = SGDRegressor
    scoring_function = r2_score
    scoring_function_name = 'r2_score'
    if y_dtype == DATA_TYPES.CATEGORICAL:
        model_cls = SGDClassifier
        scoring_function = lambda y_true, y_pred: f1_score(y_true, y_pred,
                                                           average='weighted')
        scoring_function_name = 'f1_score'

    x_scores = {}

    for x_col_name in columns.columns:
        if x_col_name == col_name:
            continue
        x_col = columns[x_col_name].values
        x_dtype = stats[x_col_name]['data_type']
        x_subtype = stats[x_col_name]['data_subtype']

        if not is_valid_x(x_col, x_dtype, x_subtype):
            continue

        x_col = x_col[non_null_index]
        x_col = x_col.reshape(-1, 1)

        x_col = preprocess_x(x_col, x_dtype)
        score = obtain_score(x_col, y_col, model_cls, scoring_function)

        x_scores[x_col_name] = score
        if score > max_predictive_power:
            max_predictive_power = score
            max_predictive_power_col = x_col_name
    predictive_combination = [col_name for col_name, score in x_scores.items()
                              if score >= combination_threshold]

    if max_predictive_power_col is None:
        predictive_power_score = 10
    else:
        predictive_power_score = to_user_friendly_score(max_predictive_power,
                                                    metric=scoring_function_name)

    return {
        'max_predictive_power': max_predictive_power,
        'max_predictive_power_col': max_predictive_power_col,
        'predictive_power_score': predictive_power_score,
        'predictive_combination': predictive_combination,
        'predictive_power_score_description': 'A high value for this score means that two of your columns can be inferred from eachother. This is done by trying to predict one column using the other via a simple linear model.',
    }
