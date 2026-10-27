"""Реальные fit и сериализация для воспроизведённых ошибок LightGBM."""
from dataclasses import replace
import os
from pathlib import Path
import pickle
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from contracts import ExperimentSpec, TaskSpec
from data_io import profile_data
from models import train_model
from proposal_models import custom_lgb
from runtime import Budget, Runner
from task_reader import load_task
from fixtures import make_task


class JudgesLightGBMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(29)
        values = rng.normal(size=(200, 3))
        frame = pd.DataFrame(values, columns=['a:b', 'a b', 'a_b'])
        frame['k[x]'] = np.where(values[:, 0] > 0, 'positive', 'negative')
        frame.loc[::17, 'a:b'] = np.nan
        frame.loc[::19, 'k[x]'] = np.nan
        self.train, self.test = frame.iloc[:160].copy(), frame.iloc[160:].copy()
        self.test.loc[self.test.index[:3], 'k[x]'] = 'unseen'
        self.y = 10 + 3*values[:160, 0] + values[:160, 1]
        self.task = TaskSpec(str(self.root/'task.md'), 'train.csv', 'test.csv', 'sample.csv',
                             'dictionary.csv', 'target', 'regression', 'rmse', ['id'],
                             frame.columns.tolist(), ['k[x]'], ['id', 'target'], 'value', [])
        self.config = ExperimentSpec(model='lightgbm', iterations=50, min_child_samples=10,
                                     missing_indicators=True, frequency_encoding=True)
        self.tr, self.va = np.arange(120), np.arange(120, 160)

    def tearDown(self):
        self.temp.cleanup()

    def fit(self, task=None, train=None, test=None, y=None):
        return train_model(task or self.task, self.config, {},
                           self.train if train is None else train,
                           self.test if test is None else test,
                           self.y if y is None else y, self.tr, self.va, 1,
                           refit=True, fixed_iterations=50, time_limit=30)

    def test_special_and_colliding_names_match_same_data_with_safe_names(self):
        before = self.train.copy(deep=True)
        bundle, valid, predicted, details = self.fit()
        mapping = dict(zip(self.train.columns, ['x1', 'x2', 'x3', 'cat']))
        safe_task = replace(self.task, feature_columns=list(mapping.values()), categorical_columns=['cat'])
        _, expected_valid, expected_test, _ = self.fit(safe_task, self.train.rename(columns=mapping),
                                                       self.test.rename(columns=mapping))
        np.testing.assert_allclose(valid, expected_valid, atol=1e-12, rtol=0)
        np.testing.assert_allclose(predicted, expected_test, atol=1e-12, rtol=0)
        pd.testing.assert_frame_equal(before, self.train)
        self.assertEqual(bundle['model'].n_estimators_, 50)
        self.assertTrue(set(details['feature_importance']).issubset(bundle['builder'].output_columns))
        self.assertIn('a:b', details['feature_importance'])

    def test_regression_serialization_preserves_predictions_with_unknown_categories(self):
        bundle, _, predicted, _ = self.fit()
        restored = pickle.loads(pickle.dumps(bundle, protocol=4))
        rebuilt = restored['model'].predict(restored['builder'].transform(self.test, categorical_codes=True))
        np.testing.assert_allclose(rebuilt, predicted, atol=1e-12, rtol=0)
        self.assertTrue(np.isfinite(predicted).all())

    def test_multiclass_probabilities_and_axis_survive_unsafe_feature_names(self):
        task = replace(self.task, problem_type='classification', metric='macro_f1',
                       classes=['high', 'low', 'middle'], prediction_kind='label')
        y = np.where(self.y > 11, 0, np.where(self.y < 9, 1, 2))
        bundle, valid, predicted, _ = self.fit(task=task, y=y)
        self.assertEqual(predicted.shape, (len(self.test), 3))
        np.testing.assert_allclose(predicted.sum(axis=1), 1, atol=1e-12)
        np.testing.assert_array_equal(bundle['model'].classes_, np.arange(3))
        self.assertTrue(np.isfinite(valid).all())

    def test_huber_reserve_uses_same_safe_feature_contract(self):
        bundle, valid, predicted, details = custom_lgb(
            self.task, self.config, {}, self.train, self.test, self.y,
            self.tr, self.va, 1, True, 50, time.monotonic()+30, 'huber')
        rebuilt = bundle['model'].predict(bundle['builder'].transform(self.test, categorical_codes=True))
        np.testing.assert_allclose(rebuilt, predicted, atol=1e-12, rtol=0)
        self.assertTrue(np.isfinite(valid).all())
        self.assertEqual(details['best_iterations'], 50)

    def test_actual_worker_without_home_keeps_lightgbm_import_and_predictions(self):
        path = make_task(self.root/'worker')
        for name in ('train.csv', 'test.csv'):
            p = path.parent/name
            frame = pd.read_csv(p, dtype={'id': str}).rename(columns={'x': 'a:b', 'category': 'k[x]'})
            frame.to_csv(p, index=False)
        p = path.parent/'data_dictionary.csv'
        frame = pd.read_csv(p)
        frame['feature'] = frame['feature'].replace({'x': 'a:b', 'category': 'k[x]'})
        frame.to_csv(p, index=False)
        task, train, test, _, dictionary = load_task(path)
        profile, _, _ = profile_data(task, train.iloc[:180], test, dictionary)
        split = path.parent/'split.npz'
        np.savez_compressed(split, train=np.arange(180), valid=np.arange(180, len(train)))
        runner = Runner(path.parent/'worker_runs', task, profile, Budget(time.monotonic(), 60, 1, 2048))
        minimal = {k: v for k, v in os.environ.items()
                   if k.upper() in {'PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'TEMP', 'TMP', 'LD_LIBRARY_PATH'}}
        with patch.dict(os.environ, minimal, clear=True):
            result = runner.run(self.config, split, 'confirm', 30, refit=True, fixed_iterations=50)
        self.assertEqual(result['status'], 'ok', result)
        self.assertEqual(result['best_iterations'], 50)
        with np.load(Path(result['directory'])/'predictions.npz') as values:
            self.assertEqual(len(values['test']), len(test))
            self.assertTrue(np.isfinite(values['test']).all())
        self.assertTrue((path.parent/'worker_runs'/'.cache'/'matplotlib').is_dir())


if __name__ == '__main__':
    unittest.main()
