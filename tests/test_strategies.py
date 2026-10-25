from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
import pandas as pd

from contracts import ExperimentSpec
from data_io import encode_target,profile_data
from fixtures import make_task
from runtime import Budget,Runner
from strategy_specs import AutoGluonSpec,validated_space,validate_semantic_transforms
from task_reader import load_task
from validation import make_splits


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_automl_contract_keeps_standard_agent_allowlist(self):
        task,*_=load_task(make_task(self.root))
        AutoGluonSpec().validate(task)
        with self.assertRaises(ValueError):
            ExperimentSpec(model='autogluon').validate(task)
        with self.assertRaises(ValueError):
            AutoGluonSpec(bag_folds=1).validate(task)
        with self.assertRaises(ValueError):
            AutoGluonSpec(target_transform='log1p').validate(task)

    def test_llm_ranges_cannot_escape_approved_bounds(self):
        ranges=validated_space({'depth':[0,50],'learning_rate':[float('nan'),1],
                                'regularization':[2,10],'leaves':[True,200]})
        self.assertEqual(ranges['depth'],[4,8])
        self.assertEqual(ranges['learning_rate'],[.025,.12])
        self.assertEqual(ranges['regularization'],[2,10])
        self.assertEqual(ranges['leaves'],[15,127])

    def test_semantic_units_and_duplicate_feature_filter(self):
        task,train,*_=load_task(make_task(self.root))
        task.feature_columns += ['period','calendar','other','existing_ratio']
        train['period']=np.arange(len(train))%5+1
        train['calendar']=1990
        train['other']=np.arange(len(train))+1
        train['existing_ratio']=train.x/train.other
        dd=pd.DataFrame({'feature':['period','calendar'],'unit':['код года','год']})
        invalid=[{'op':'difference','columns':['period','calendar']}]
        duplicate=[{'op':'ratio','columns':['x','other']}]
        valid=[{'op':'difference','columns':['x','other']}]
        self.assertEqual(validate_semantic_transforms(task,train,dd,invalid),[])
        self.assertEqual(validate_semantic_transforms(task,train,dd,duplicate),[])
        self.assertEqual(validate_semantic_transforms(task,train,dd,valid),valid)

    def test_worker_module_allowlist(self):
        task,*_=load_task(make_task(self.root))
        with self.assertRaises(ValueError):
            Runner(self.root/'runs',task,{},Budget(time.monotonic(),30,1,1024),worker_module='os')

    def test_autogluon_search_and_frozen_confirmation(self):
        try:
            installed=importlib.util.find_spec('autogluon.tabular') is not None
        except ModuleNotFoundError:
            installed=False
        if not installed:
            self.skipTest('AutoGluon присутствует только в исследовательском образе')
        for classification in (False,True):
            with self.subTest(classification=classification):
                case=self.root/str(classification)
                task,train,test,_,dd=load_task(make_task(case,classification=classification))
                y=encode_target(task,train)
                profile,groups,times=profile_data(task,train,test,dd)
                splits=make_splits(task,y,groups,times)
                jobs=case/'jobs';jobs.mkdir()
                for split in splits:
                    np.savez_compressed(jobs/f'{split.name}.npz',train=split.train,valid=split.valid)
                runner=Runner(jobs,task,profile,Budget(time.monotonic(),240,1,4096),worker_module='strategy_worker')
                config=AutoGluonSpec(iterations=60,bag_folds=2 if classification else 0,
                                    class_weight_power=.5 if classification else 0)
                search=runner.run(config,jobs/'search.npz','search',100)
                self.assertEqual(search['status'],'ok',search)
                confirm=runner.run(config,jobs/'confirm.npz','confirm',80,refit=True,
                                   job_options={'source_directory':search['directory']})
                self.assertEqual(confirm['status'],'ok',confirm)
                with np.load(Path(confirm['directory'])/'predictions.npz') as values:
                    self.assertEqual(len(values['valid']),len(splits[1].valid))
                    self.assertEqual(len(values['test']),len(test))
                np.savez_compressed(jobs/'full.npz',train=np.arange(len(train)),valid=np.array([],dtype=int))
                full=runner.run(config,jobs/'full.npz','full',80,refit=True,
                       job_options={'source_directory':search['directory'],'use_all_training':True})
                self.assertEqual(full['status'],'ok',full)
                with np.load(Path(full['directory'])/'predictions.npz') as values:
                    self.assertEqual(len(values['valid']),0)
                    self.assertEqual(len(values['test']),len(test))


if __name__=='__main__':
    unittest.main()
