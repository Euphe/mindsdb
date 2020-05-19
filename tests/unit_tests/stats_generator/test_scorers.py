import pytest
from pytest import mark, param
import numpy as np
import pandas as pd
from mindsdb.libs.phases.stats_generator.scores import \
    (compute_predictive_power_score, DATA_TYPES)


test_x_column_data = ("x_dtype,x_data", [
    param(DATA_TYPES.NUMERIC, np.linspace(0, 1000, 1000), id='x Numeric'),
    param(DATA_TYPES.CATEGORICAL, [str(x) for x in list(range(10))*100], id='x Categorical')
])


def get_stats_columns(y_dtype, x_dtype, x_data):
    stats = {
        'x': {
            'data_type': str(x_dtype)
        },
        'y': {
            'data_type': str(y_dtype)
        },
    }

    columns = pd.DataFrame({'x': x_data})
    return stats, columns


class TestPredictivePowerScore:
    @mark.parametrize(*test_x_column_data)
    def test_numeric_dependent_linear(self, x_dtype, x_data):
        """y is Numeric, y is linearly dependent on x"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           x_dtype=x_dtype,
                                           x_data=x_data)
        columns['y'] = columns.x.astype(float)*2 + 1

        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.1)
        assert output['predictive_power_score'] == 0
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    # @TODO better solution for nonlinear dependency, so that it outputs a score of 1
    @mark.parametrize(*test_x_column_data)
    def test_numeric_dependent_nonlinear(self, x_dtype, x_data):
        """y is Numeric, y is nonlinearly dependent on x"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           x_dtype=x_dtype,
                                           x_data=x_data)
        columns['y'] = columns.x.astype(float)**2 + 3

        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.1)
        assert output['predictive_power_score'] < 3
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_numeric_independent(self, x_dtype, x_data):
        """y is Numeric, y is independent of x"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           x_dtype=x_dtype,
                                           x_data=x_data)
        columns['y'] = np.random.rand(len(columns['x']))

        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 0, atol=0.2)
        assert output['predictive_power_score'] >= 8
        assert 'max_predictive_power_col' in output
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_numeric_dependent_linear_noisy(self, x_dtype, x_data):
        """y is Numeric, y is linearly dependent on x with noise"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           x_dtype=x_dtype,
                                           x_data=x_data)
        columns['y'] = columns.x.astype(float)*2 + 1 + columns.x.astype(float).mean() * np.random.rand(len(columns['x']))
        output = compute_predictive_power_score(stats, columns, 'y')
        np.testing.assert_allclose(output['max_predictive_power'], 0.9, atol=0.1)
        assert output['predictive_power_score'] <= 3
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_numeric_nan_values(self, x_dtype, x_data):
        """y is Numeric, y is linearly dependent on x and both contain nan values"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           x_dtype=x_dtype,
                                           x_data=x_data)
        columns['y'] = columns.x.astype(float)*2 + 1

        # Make 10% random values nan
        nan_index_x = np.random.choice(range(int(0.1*len(columns))), int(0.1*len(columns)))
        columns.loc[nan_index_x, ('x', )] = None

        # Different amount of choices, so that despite fixed random seed
        # nan values are not same as for x
        nan_index_y = np.random.choice(range(int(0.1*len(columns))), int(0.1*len(columns)+1))
        columns.loc[nan_index_y, ('y',)] = None

        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.3)
        assert output['predictive_power_score'] <= 1
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']

    @mark.parametrize(*test_x_column_data)
    def test_numeric_two_columns(self, x_dtype, x_data):
        """y is Numeric, y is linearly dependent on x, x1 is independent of y"""
        stats, columns = get_stats_columns(y_dtype=DATA_TYPES.NUMERIC,
                                           x_dtype=x_dtype,
                                           x_data=x_data)
        stats['x1'] = {'data_type': str(DATA_TYPES.NUMERIC)}
        columns['x1'] = np.random.rand(len(columns))
        columns['y'] = columns.x.astype(float) * 2 + 1

        output = compute_predictive_power_score(stats, columns, 'y')

        np.testing.assert_allclose(output['max_predictive_power'], 1, atol=0.1)
        assert output['predictive_power_score'] == 0
        assert output['max_predictive_power_col'] == 'x'
        assert output['predictive_power_score_description']
