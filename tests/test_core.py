from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
from types import SimpleNamespace

import numpy as np
import pandas as pd

from contracts import ExperimentSpec
from data_io import encode_target, profile_data
from llm_client import LLMPlanner, parse_json
from experiment_store import Journal
from metrics import evaluate
from models import train_model
from preprocessing import FeatureBuilder
from runtime import Budget, Runner
from submission import fallback_prediction, write_submission
from task_reader import detect_metric, load_task, read_csv
from validation import make_splits
from fixtures import make_task


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def load(self, **kwargs):
        return load_task(make_task(self.root, **kwargs))

    def test_schema_paths_and_id_precision(self):
        task, train, test, sample, dd = self.load()
        self.assertEqual(task.metric, "rmse")
        self.assertEqual(task.target, "response")
        self.assertEqual(test.id.iloc[0], "000240")
        self.assertNotIn("id", task.feature_columns)
        self.assertEqual(task.output_columns, ["id", "response"])

    def test_primary_metric_over_reference(self):
        self.assertEqual(detect_metric("Пример RMSE.\nОсновная метрика — Macro F1.\nДополнительно Accuracy"), "macro_f1")
        self.assertEqual(detect_metric("Evaluation metric: RMSLE"), "rmsle")
        self.assertEqual(detect_metric("Primary metric: balanced accuracy"), "balanced_accuracy")
        self.assertIsNone(detect_metric("Новая неизвестная метрика"))

    def test_sample_identifies_nonstandard_id_without_dictionary(self):
        path = make_task(self.root)
        for name in ('train.csv', 'test.csv', 'sample_submission.csv'):
            frame = pd.read_csv(self.root/name, dtype={'id': str}).rename(columns={'id': 'record_key','x':'index'})
            frame.to_csv(self.root/name, index=False)
        (self.root/'data_dictionary.csv').unlink()
        task, _, test, *_ = load_task(path)
        self.assertEqual(task.id_columns, ['record_key'])
        self.assertNotIn('record_key', task.feature_columns)
        self.assertIn('index', task.feature_columns)
        self.assertEqual(test.record_key.iloc[0], '000240')

    def test_identifier_can_be_absent_from_output(self):
        path = make_task(self.root)
        pd.read_csv(self.root/'sample_submission.csv').drop(columns=['id']).to_csv(self.root/'sample_submission.csv', index=False)
        task, train, test, *_ = load_task(path)
        self.assertEqual(task.id_columns, [])
        self.assertNotIn('id', task.feature_columns)
        write_submission(task, test, np.ones(len(test)), self.root/'submission.csv')
        self.assertEqual(pd.read_csv(self.root/'submission.csv').columns.tolist(), ['response'])

    def test_duplicate_csv_header_is_rejected(self):
        path = self.root/"bad.csv"
        path.write_text("x,x\n1,2\n")
        with self.assertRaises(ValueError):
            read_csv(path)

    def test_config_does_not_accept_code_or_target_features(self):
        task, *_ = self.load()
        for transform in [{"op": "eval", "columns": ["x"]},
                          {"op": "ratio", "columns": ["response", "x"]},
                          {"op": "log1p", "columns": ["x"], "code": "print(1)"}]:
            with self.assertRaises(ValueError):
                replace(ExperimentSpec(), transforms=[transform]).validate(task)
        with self.assertRaises(ValueError):
            replace(ExperimentSpec(), iterations=100.5).validate(task)
        with self.assertRaises(TypeError):
            ExperimentSpec.from_dict({"shell_command": "echo hello"}, task)

    def test_same_effective_model_is_not_experimented_twice(self):
        base = ExperimentSpec(model='lightgbm')
        self.assertEqual(base.key, replace(base, depth=9, hypothesis='Другая формулировка').key)
        self.assertNotEqual(base.key, replace(base, leaves=63).key)

    def test_training_only_frequency_and_unknown_category(self):
        task, train, test, *_ = self.load()
        builder = FeatureBuilder(task, replace(ExperimentSpec(), frequency_encoding=True), {}).fit(train)
        out = builder.transform(test)
        self.assertEqual(out["category__frequency"].iloc[0], 0)
        categorical = builder.transform(test, categorical_codes=True)
        self.assertTrue(pd.isna(categorical.category.iloc[0]))
        self.assertEqual(out.columns.tolist(), builder.output_columns)

    def test_invalid_log_and_safe_division(self):
        task, train, test, *_ = self.load()
        bad = replace(ExperimentSpec(), transforms=[{"op": "log1p", "columns": ["x"]}])
        with self.assertRaises(ValueError):
            FeatureBuilder(task, bad, {}).fit(train)
        config = replace(ExperimentSpec(), transforms=[{"op": "ratio", "columns": ["x", "x"]}])
        train.loc[0, "x"] = 0
        builder = FeatureBuilder(task, config, {}).fit(train)
        transformed = builder.transform(train)
        self.assertTrue(np.isnan(transformed["__derived_0_ratio"].iloc[0]))

    def test_temporal_search_and_closed_confirmation(self):
        task, train, test, _, dd = self.load(temporal=True)
        profile, groups, times = profile_data(task, train, test, dd)
        splits = make_splits(task, encode_target(task, train), groups, times)
        self.assertEqual(profile["time_column"], "period_year")
        self.assertEqual(set(times[splits[0].train]), {1, 2, 3})
        self.assertEqual(set(times[splits[0].valid]), {4})
        self.assertEqual(set(times[splits[1].valid]), {5})
        self.assertFalse(set(splits[1].valid) & set(np.r_[splits[0].train, splits[0].valid]))

    def test_unique_timestamps_use_blocks_not_one_row(self):
        path = make_task(self.root)
        dates = pd.date_range('2020-01-01',periods=290,freq='h')
        for name,start in [('train.csv',0),('test.csv',240)]:
            frame = pd.read_csv(self.root/name,dtype={'id':str})
            frame['event_time'] = dates[start:start+len(frame)].astype(str)
            frame.to_csv(self.root/name,index=False)
        task,train,test,_,dd = load_task(path)
        profile,groups,times = profile_data(task,train,test,dd)
        self.assertEqual(profile['time_column'],'event_time')
        splits = make_splits(task,encode_target(task,train),groups,times)
        self.assertEqual(len(splits[0].valid),48)
        self.assertEqual(len(splits[1].valid),48)
        self.assertLess(times[splits[0].train].max(),times[splits[0].valid].min())
        self.assertFalse(set(splits[1].valid) & set(np.r_[splits[0].train,splits[0].valid]))

    def test_grouped_classification_has_no_overlap(self):
        task, train, test, _, dd = self.load(classification=True)
        # Две копии каждой строки принадлежат одной группе независимо от id.
        repeated = pd.concat([train, train], ignore_index=True)
        repeated["id"] = [str(i) for i in range(len(repeated))]
        profile, groups, times = profile_data(task, repeated, test, dd)
        splits = make_splits(task, encode_target(task, repeated), groups, times)
        for split in splits:
            self.assertFalse(set(groups[split.train]) & set(groups[split.valid]))
        self.assertFalse(set(splits[1].valid) & set(np.r_[splits[0].train, splits[0].valid]))

    def test_macro_f1_includes_all_classes(self):
        task, train, *_ = self.load(classification=True)
        y = np.array([0, 0, 1, 1, 2, 2])
        prediction = np.zeros((6, 3)); prediction[:, 0] = 1
        result = evaluate(task, y, prediction)
        self.assertAlmostEqual(result["macro_f1"], 1/6)
        self.assertEqual(len(result["per_class_f1"]), 3)

    def test_submission_atomicity_and_ids(self):
        task, train, test, *_ = self.load()
        path = self.root/"submission.csv"
        write_submission(task, test, np.full(len(test), 3.), path)
        previous = path.read_bytes()
        with self.assertRaises(ValueError):
            write_submission(task, test, np.full(len(test), np.nan), path)
        self.assertEqual(path.read_bytes(), previous)
        reopened = pd.read_csv(path, dtype={"id": str})
        self.assertEqual(reopened.id.tolist(), test.id.tolist())

    def test_probability_column_order(self):
        task, train, test, *_ = self.load(classification=True, probabilities=True)
        self.assertEqual(task.prediction_kind, "probabilities")
        prediction = np.tile([.1, .2, .7], (len(test), 1))
        path = self.root/"submission.csv"
        write_submission(task, test, prediction, path)
        result = pd.read_csv(path)
        for c in task.probability_columns:
            self.assertAlmostEqual(result[c].iloc[0], prediction[0, task.classes.index(c)])

    def test_prefixed_probability_columns(self):
        path = make_task(self.root, classification=True, probabilities=True)
        sample = pd.read_csv(self.root/'sample_submission.csv', dtype={'id': str})
        sample.rename(columns={c:'prob_'+c for c in sample if c!='id'}).to_csv(self.root/'sample_submission.csv', index=False)
        task, _, test, *_ = load_task(path)
        self.assertEqual(task.prediction_kind, 'probabilities')
        prediction = np.tile([.1,.2,.7], (len(test),1))
        write_submission(task,test,prediction,self.root/'submission.csv')
        result = pd.read_csv(self.root/'submission.csv')
        self.assertAlmostEqual(result.prob_middle.iloc[0], prediction[0, task.classes.index('middle')])

    def test_explicit_positive_class_controls_probability_output(self):
        path = make_task(self.root, classification=True)
        train = pd.read_csv(self.root/'train.csv', dtype={'id':str})
        train['category_to_predict'] = (train['category_to_predict']=='high').astype(int)
        train.to_csv(self.root/'train.csv', index=False)
        path.write_text('Основная метрика — ROC AUC.\nPositive class: `0`.\n', encoding='utf-8')
        task, _, test, *_ = load_task(path)
        self.assertEqual(task.positive_index, 0)
        self.assertEqual(task.prediction_kind, 'probability')
        prediction = np.tile([.8,.2], (len(test),1))
        write_submission(task,test,prediction,self.root/'submission.csv')
        self.assertAlmostEqual(pd.read_csv(self.root/'submission.csv').category_to_predict.iloc[0], .8)

    def test_json_rejects_trailing_code(self):
        self.assertEqual(parse_json('```json\n{"model":"catboost"}\n```')["model"], "catboost")
        self.assertEqual(parse_json('<think>hidden</think>{"model":"linear"}')["model"], "linear")
        with self.assertRaises(ValueError):
            parse_json('{}\n__import__("os")')

    def test_both_backends_train_and_predict_unseen_categories(self):
        for model, classification in [("catboost", False), ("catboost", True), ("lightgbm", True), ("linear", False)]:
            with self.subTest(model=model):
                task, train, test, _, dd = self.load(classification=classification)
                y = encode_target(task, train)
                profile, groups, times = profile_data(task, train, test, dd)
                split = make_splits(task, y, groups, times)[0]
                config = ExperimentSpec(model=model, iterations=100, depth=4, learning_rate=.1)
                bundle, val, pred, details = train_model(task, config, profile, train, test,
                                                         y, split.train, split.valid, 1)
                score = evaluate(task, y[split.valid], val)[task.metric]
                dummy = evaluate(task, y[split.valid], fallback_prediction(task, y[split.train], len(split.valid)))[task.metric]
                self.assertTrue(score > dummy if task.maximize else score < dummy)
                self.assertEqual(len(pred), len(test))
                self.assertGreater(details["best_iterations"], 0)

    def test_soft_time_limit_keeps_search_model_but_rejects_partial_refit(self):
        task, train, test, _, dd = self.load()
        y = encode_target(task,train)
        profile,groups,times = profile_data(task,train,test,dd)
        split = make_splits(task,y,groups,times)[0]
        for backend in ('catboost','lightgbm'):
            config = ExperimentSpec(model=backend,iterations=6000)
            _, valid, prediction, details = train_model(task,config,profile,train,test,y,
                split.train,split.valid,1,time_limit=.00001)
            self.assertTrue(details['training_stopped_for_budget'])
            self.assertTrue(np.isfinite(prediction).all())
            self.assertLess(details['best_iterations'],6000)
            with self.assertRaises(TimeoutError):
                train_model(task,config,profile,train,test,y,split.train,np.array([],dtype=int),1,
                            refit=True,fixed_iterations=6000,time_limit=.00001)

    def test_text_pipeline_uses_tfidf_without_pretrained_weights(self):
        path = make_task(self.root,classification=True)
        for name in ('train.csv','test.csv'):
            frame = pd.read_csv(self.root/name,dtype={'id':str})
            words = np.where(frame.x>.5,'excellent',np.where(frame.x<-.5,'terrible','ordinary'))
            frame['document'] = [f'This is a sufficiently long document about an {word} experience number {i}'
                                 for i,word in enumerate(words)]
            frame.drop(columns=['x','category']).to_csv(self.root/name,index=False)
        task,train,test,_,dd = load_task(path)
        profile,groups,times = profile_data(task,train,test,dd)
        self.assertIn('document',profile['text_columns'])
        y = encode_target(task,train)
        split = make_splits(task,y,groups,times)[0]
        _,valid,prediction,_ = train_model(task,ExperimentSpec(model='linear',regularization=1),
             profile,train,test,y,split.train,split.valid,1)
        self.assertGreater(evaluate(task,y[split.valid],valid)['macro_f1'],.8)
        np.testing.assert_allclose(prediction.sum(axis=1),1,atol=1e-7)

    def test_llm_protocol_and_empty_response_accounting(self):
        task,train,test,_,dd = self.load(classification=True)
        profile,_,_ = profile_data(task,train,test,dd)
        client = MagicMock()
        client.__enter__.return_value = client
        usage = SimpleNamespace(total_tokens=111,completion_tokens=20)
        client.chat.completions.create.return_value = SimpleNamespace(usage=usage,choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"class_weight_power":0.5}'),finish_reason='stop')])
        planner = LLMPlanner(Journal(self.root/'journal'),Budget(time.monotonic(),300,1,1024),2)
        planner.key = 'test-only-secret'
        with patch('openai.OpenAI',return_value=client), patch.dict(os.environ,{'OPENAI_MODEL':'qwen/qwen3.8-27b'}):
            self.assertIsNotNone(planner.propose(task,profile,[]))
            kwargs = client.chat.completions.create.call_args.kwargs
            self.assertFalse(kwargs['extra_body']['reasoning']['enabled'])
            self.assertTrue(kwargs['messages'][-1]['content'].endswith('/no_think'))
            client.chat.completions.create.return_value.choices[0].message.content = None
            self.assertIsNone(planner.propose(task,profile,[]))
        self.assertEqual(planner.tokens,222)
        self.assertNotIn('test-only-secret',(self.root/'journal'/'experiments.jsonl').read_text(encoding='utf-8'))

    def test_watchdog_terminates_stalled_process(self):
        task, *_ = self.load()
        fake = self.root/"fake"; fake.mkdir()
        (fake/"worker.py").write_text("def main(path):\n import time\n time.sleep(20)\n return 0\n")
        runner = Runner(self.root/"jobs", task, {}, Budget(time.monotonic(), 30, 1, 128))
        runner.code_root = fake
        started = time.monotonic()
        result = runner.run(ExperimentSpec(), self.root/"unused.npz", "test", .5)
        self.assertEqual(result["status"], "timeout")
        self.assertLess(time.monotonic()-started, 5)

    def test_worker_environment_does_not_include_api_key(self):
        task, *_ = self.load()
        fake = self.root/"fake"; fake.mkdir()
        (fake/"worker.py").write_text(
            "def main(path):\n import os,json,pathlib\n"
            " assert 'OPENAI_API_KEY' not in os.environ\n"
            " pathlib.Path(path).with_name('result.json').write_text(json.dumps({'status':'ok'}))\n return 0\n")
        runner = Runner(self.root/"jobs", task, {}, Budget(time.monotonic(), 30, 1, 128))
        runner.code_root = fake
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-not-a-real-key"}):
            result = runner.run(ExperimentSpec(), self.root/"unused.npz", "test", 4)
        self.assertEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
