import pytest
from datetime import datetime, timedelta
from pytest import mark, param
import numpy as np
import pandas as pd
from mindsdb.libs.phases.stats_generator.scores import \
    (compute_predictive_power_score, DATA_TYPES, DATA_SUBTYPES)


test_x_column_data = ("x_dtype,x_subtype,x_data", [
    param(DATA_TYPES.NUMERIC, DATA_SUBTYPES.FLOAT, np.linspace(0, 1000, 1000), id='x Numeric'),
    param(DATA_TYPES.CATEGORICAL, DATA_SUBTYPES.MULTIPLE, [str(x) for x in list(range(10))*100], id='x Categorical str'),
    param(DATA_TYPES.CATEGORICAL, DATA_SUBTYPES.MULTIPLE, [x for x in list(range(10))*100], id='x Categorical int'),
])


def get_stats_columns(y_dtype, y_subtype, x_dtype, x_subtype, x_data):
    stats = {
        'x': {
            'data_type': str(x_dtype),
            'data_subtype': str(x_subtype)
        },
        'y': {
            'data_type': str(y_dtype),
            'data_subtype': str(y_subtype)
        },
    }

    columns = pd.DataFrame({'x': x_data})
    return stats, columns


class TestPredictivePowerScore:
    @mark.parametrize(*test_x_column_data)
    def test_numeric_dependent_linear(self, x_dtype, x_subtype, x_data):
        """y is Numeric, y is linearly dependent on x"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           y_subtype=DATA_SUBTYPES.FLOAT,
                                           x_dtype=x_dtype,
                                           x_subtype=x_subtype,
                                           x_data=x_data)
        columns['y'] = columns.x.astype(float)*2 + 1

        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.1)
        assert output['predictive_power_score'] == 0
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    def test_numeric_x_timestamp(self):
        """y is Numeric, x is a timestamp, y is linearly dependent on x"""
        x_data = [(datetime.now() - timedelta(minutes=int(i))).timestamp() for i in range(1000)]
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           y_subtype=DATA_SUBTYPES.FLOAT,
                                           x_dtype=DATA_TYPES.DATE,
                                           x_subtype=DATA_SUBTYPES.TIMESTAMP,
                                           x_data=x_data)
        columns['y'] = columns.x.astype(float)*2 + 10

        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.1)
        assert output['predictive_power_score'] == 0
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_numeric_dependent_nonlinear(self, x_dtype, x_subtype, x_data):
        """y is Numeric, y is nonlinearly dependent on x"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           y_subtype=DATA_SUBTYPES.FLOAT,
                                           x_dtype=x_dtype,
                                           x_subtype=x_subtype,
                                           x_data=x_data)
        columns['y'] = columns.x.astype(float)**2 + 3

        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.2)
        assert output['predictive_power_score'] <= 3
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_numeric_independent(self, x_dtype, x_subtype, x_data):
        """y is Numeric, y is independent of x"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           y_subtype=DATA_SUBTYPES.FLOAT,
                                           x_dtype=x_dtype,
                                           x_subtype=x_subtype,
                                           x_data=x_data)
        columns['y'] = np.random.rand(len(columns['x']))

        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 0, atol=0.2)
        assert output['predictive_power_score'] >= 8
        assert 'max_predictive_power_col' in output
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_numeric_dependent_linear_noisy(self, x_dtype, x_subtype, x_data):
        """y is Numeric, y is linearly dependent on x with noise"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           y_subtype=DATA_SUBTYPES.FLOAT,
                                           x_dtype=x_dtype,
                                           x_subtype=x_subtype,
                                           x_data=x_data)
        columns['y'] = columns.x.astype(float)*2 + 1 + columns.x.astype(float).mean() * np.random.rand(len(columns['x']))
        output = compute_predictive_power_score(stats, columns, 'y')
        np.testing.assert_allclose(output['max_predictive_power'], 0.9, atol=0.1)
        assert output['predictive_power_score'] <= 3
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_numeric_nan_values(self, x_dtype, x_subtype, x_data):
        """y is Numeric, y is linearly dependent on x and both contain nan values"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           y_subtype=DATA_SUBTYPES.FLOAT,
                                           x_dtype=x_dtype,
                                           x_subtype=x_subtype,
                                           x_data=x_data)
        columns['y'] = columns.x.astype(float)*2 + 1

        # Make 5% random values nan
        nan_index_x = np.random.choice(range(int(0.05*len(columns))), int(0.05*len(columns)))
        columns.loc[nan_index_x, ('x', )] = None

        # Different amount of choices, so that despite fixed random seed
        # nan values are not same as for x
        nan_index_y = np.random.choice(range(int(0.05*len(columns))), int(0.05*len(columns)+1))
        columns.loc[nan_index_y, ('y',)] = None

        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.3)
        assert output['predictive_power_score'] <= 4
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_numeric_two_columns(self, x_dtype, x_subtype, x_data):
        """y is Numeric, y is linearly dependent on x, x1 is independent of y"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           y_subtype=DATA_SUBTYPES.FLOAT,
                                           x_dtype=x_dtype,
                                           x_subtype=x_subtype,
                                           x_data=x_data)
        stats['x1'] = {'data_type': str(DATA_TYPES.NUMERIC),
                       'data_subtype': str(DATA_SUBTYPES.FLOAT)}
        columns['x1'] = np.random.rand(len(columns))
        columns['y'] = columns.x.astype(float) * 2 + 1

        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.1)
        assert output['predictive_power_score'] == 0
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_categorical_dependent(self, x_dtype, x_subtype, x_data):
        """y is Categorical, y is dependent on x"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.CATEGORICAL,
                                           y_subtype=DATA_SUBTYPES.MULTIPLE,
                                           x_dtype=x_dtype,
                                           x_subtype=x_subtype,
                                           x_data=x_data)
        # Divide all x values in 3 approx equal sized bins
        num_bins = 3
        bin_width = columns.x.astype(float).max() / num_bins
        columns['y'] = np.round(columns.x.astype(float) // (bin_width+1))
        output = compute_predictive_power_score(stats, columns, 'y')
        np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.2)
        assert output['predictive_power_score'] <= 2
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_categorical_independent(self, x_dtype, x_subtype, x_data):
        """y is Categorical, y independent of x"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.CATEGORICAL,
                                           y_subtype=DATA_SUBTYPES.MULTIPLE,
                                           x_dtype=x_dtype,
                                           x_subtype=x_subtype,
                                           x_data=x_data)
        columns['y'] = np.random.choice(range(10), len(columns))
        output = compute_predictive_power_score(stats, columns, 'y')
        np.testing.assert_allclose(output['max_predictive_power'], 0, atol=0.1)
        assert output['predictive_power_score'] >= 9
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_categorical_nan_values(self, x_dtype, x_subtype, x_data):
        """y is Categorical, y is dependent on x and both contain nan values"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.CATEGORICAL,
                                           y_subtype=DATA_SUBTYPES.MULTIPLE,
                                           x_dtype=x_dtype,
                                           x_subtype=x_subtype,
                                           x_data=x_data)
        # Divide all x values in 3 approx equal sized bins
        num_bins = 3
        bin_width = columns.x.astype(float).max() / num_bins
        columns['y'] = np.round(columns.x.astype(float) // (bin_width + 1))

        nan_index_x = np.random.choice(range(int(0.05 * len(columns))),
                                       int(0.05 * len(columns)))
        columns.loc[nan_index_x, ('x',)] = None

        nan_index_y = np.random.choice(range(int(0.05 * len(columns))),
                                       int(0.05 * len(columns) + 1))
        columns.loc[nan_index_y, ('y',)] = None

        output = compute_predictive_power_score(stats, columns, 'y')
        np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.6)
        assert output['predictive_power_score'] <= 4
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_categorical_two_columns(self, x_dtype, x_subtype, x_data):
        """y is Categorical, y is dependent on x, x1 is independent of y"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.CATEGORICAL,
                                           y_subtype=DATA_SUBTYPES.MULTIPLE,
                                           x_dtype=x_dtype,
                                           x_subtype=x_subtype,
                                           x_data=x_data)
        # Divide all x values in 3 approx equal sized bins
        num_bins = 3
        bin_width = columns.x.astype(float).max() / num_bins
        columns['y'] = np.round(columns.x.astype(float) // (bin_width + 1))

        stats['x1'] = {'data_type': str(DATA_TYPES.CATEGORICAL),
                       'data_subtype': str(DATA_SUBTYPES.MULTIPLE)}
        columns['x1'] = np.random.choice(range(10), len(columns))

        output = compute_predictive_power_score(stats, columns, 'y')
        np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.2)
        assert output['predictive_power_score'] <= 2
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_y_date(self, x_dtype, x_subtype, x_data):
        """y is Date"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.DATE,
                                           y_subtype=DATA_SUBTYPES.DATE,
                                           x_dtype=x_dtype,
                                           x_subtype=x_subtype,
                                           x_data=x_data)
        stats['y']['data_subtype'] = DATA_SUBTYPES.DATE
        columns['y'] = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(columns))]
        output = compute_predictive_power_score(stats, columns, 'y')

        assert 'max_predictive_power' not in output
        assert 'predictive_power_score' not in output
        assert 'max_predictive_power_col' not in output
        assert 'predictive_power_score_description' in output
        assert output['predictive_power_score_description'] == f'Predictive power score for ' \
            f'data type {DATA_TYPES.DATE} not supported'

    @mark.parametrize(*test_x_column_data)
    def test_y_timestamp(self, x_dtype, x_subtype, x_data):
        """y is Timestamp, y is linearly dependent on x"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.DATE,
                                           y_subtype=DATA_SUBTYPES.TIMESTAMP,
                                           x_dtype=x_dtype,
                                           x_subtype=x_subtype,
                                           x_data=x_data)
        columns['y'] = [
            pd.to_datetime(datetime.now() - timedelta(seconds=int(columns.x.values[i]))).value for i in range(len(columns))]
        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.1)
        assert output['predictive_power_score'] <= 1
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    def test_y_unsupported_type(self):
        """y type is not supported"""
        for dtype in (DATA_TYPES.FILE_PATH,
                      DATA_TYPES.URL,
                      DATA_TYPES.SEQUENTIAL):
            stats, columns = get_stats_columns(y_dtype=dtype,
                                               y_subtype=DATA_SUBTYPES.FLOAT,
                                               x_dtype=DATA_TYPES.NUMERIC,
                                               x_subtype=DATA_SUBTYPES.FLOAT,
                                               x_data=[1, 2])
            columns['y'] = [1, 3]

            output = compute_predictive_power_score(stats, columns, 'y')

            assert 'max_predictive_power' not in output
            assert 'predictive_power_score' not in output
            assert 'max_predictive_power_col' not in output
            assert 'predictive_power_score_description' in output
            assert output[
                       'predictive_power_score_description'] == f'Predictive power score for ' \
                       f'data type {dtype} not supported'

    def test_y_has_one_value(self):
        """y has only one non-null value"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           y_subtype=DATA_SUBTYPES.FLOAT,
                                           x_dtype=DATA_TYPES.NUMERIC,
                                           x_subtype=DATA_SUBTYPES.FLOAT,
                                           x_data=[1, 2])
        columns['y'] = [1, 1]

        output = compute_predictive_power_score(stats, columns, 'y')

        assert 'max_predictive_power' not in output
        assert 'predictive_power_score' not in output
        assert 'max_predictive_power_col' not in output
        assert 'predictive_power_score_description' in output
        assert output[
                   'predictive_power_score_description'] == f'Column contains only a single value, ' \
            f'predictive power score not supported'

    def test_x_not_supported(self):
        """no x column is of supported type"""
        for dtype in (DATA_TYPES.FILE_PATH,
                      DATA_TYPES.URL,
                      DATA_TYPES.SEQUENTIAL):
            stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                               y_subtype=DATA_SUBTYPES.FLOAT,
                                               x_dtype=dtype,
                                               x_subtype=DATA_SUBTYPES.IMAGE,
                                               x_data=['1', '2'])

            columns['y'] = [1, 3]

            output = compute_predictive_power_score(stats, columns, 'y')
            assert output['max_predictive_power'] == 0
            assert output['predictive_power_score'] == 10
            assert output['max_predictive_power_col'] == None
            assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_x_unsupported_and_supported_type(self, x_dtype, x_subtype, x_data):
        """x columns are a mix of supported and unsupported types"""
        for dtype in (DATA_TYPES.FILE_PATH,
                      DATA_TYPES.URL,
                      DATA_TYPES.SEQUENTIAL,
                      DATA_TYPES.DATE):
            stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                               y_subtype=DATA_SUBTYPES.FLOAT,
                                               x_dtype=x_dtype,
                                               x_subtype=x_subtype,
                                               x_data=x_data)

            stats['x1'] = {'data_type': dtype, 'data_subtype': DATA_SUBTYPES.DATE}
            columns['x1'] = [f'val{i}' for i in range(len(columns))]
            columns['y'] = columns.x.astype(float)+1

            output = compute_predictive_power_score(stats, columns, 'y')

            np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.1)
            assert output['predictive_power_score'] <= 1
            assert output['max_predictive_power_col'] == 'x'
            assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_x_one_value(self, x_dtype, x_subtype, x_data):
        """x columns include a column that consists of just one value"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           y_subtype=DATA_SUBTYPES.FLOAT,
                                           x_dtype=x_dtype,
                                           x_subtype=x_subtype,
                                           x_data=x_data)

        stats['x1'] = {'data_type': DATA_TYPES.NUMERIC, 'data_subtype': DATA_SUBTYPES.FLOAT}
        columns['x1'] = [0 for i in range(len(columns))]
        columns['y'] = columns.x.astype(float) + 1

        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 1,
                                   atol=0.1)
        assert output['predictive_power_score'] <= 1
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    def test_numeric_predictive_combination(self):
        """y is Numeric, y is dependent on x + x1, independent of x2"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           y_subtype=DATA_SUBTYPES.FLOAT,
                                           x_dtype=DATA_TYPES.NUMERIC,
                                           x_subtype=DATA_SUBTYPES.FLOAT,
                                           x_data=np.linspace(0, 1000, 1000))
        stats['x1'] = stats['x']
        stats['x2'] = stats['x']
        columns['x2'] = np.random.rand(len(columns))

        columns['x1'] = columns.x.astype(float).mean()*np.random.rand(len(columns))
        columns['y'] = 0.5*columns.x.astype(float) + columns.x1

        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 0.5, atol=0.3)
        assert output['predictive_power_score'] <= 8
        assert output['predictive_power_score_description']
        assert output['predictive_combination'] == ['x', 'x1']
