"""Обязательная ранняя линейная семья и содержательные уточнения лидера."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from contracts import ExperimentSpec
from fixtures import make_task
from research import Research
from task_reader import load_task


class LinearPortfolioTests(unittest.TestCase):
    def test_numeric_tasks_always_start_with_linear_before_any_llm(self):
        for classification in (False,True):
            with self.subTest(classification=classification),tempfile.TemporaryDirectory() as tmp:
                task,*_=load_task(make_task(Path(tmp),classification=classification))
                planner=Mock();planner.available=True
                research=Research(task,{"minority_fraction":.01},planner)
                first=research.next([])
                self.assertEqual(first.model,"linear")
                self.assertEqual(first.regularization,5)
                self.assertEqual(first.class_weight_power,.5 if classification else 0.)
                planner.propose.assert_not_called()
                self.assertTrue(research.essential_pending)

    def test_original_controls_must_be_attempted_before_plateau_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            task,*_=load_task(make_task(Path(tmp),classification=True))
        planner=Mock();planner.available=False
        research=Research(task,{"minority_fraction":.01,"feature_count":2,"train_rows":144},
                          planner,include_alternatives=True)
        delivered=[]
        while research.essential_pending:
            config=research.next([])
            self.assertIsNotNone(config)
            delivered.append(config)
            self.assertLess(len(delivered),15)
        self.assertEqual(delivered[0].model,"linear")
        self.assertTrue(any(c.model=="extra_trees" for c in delivered))
        self.assertTrue(any(c.model=="lightgbm" and c.class_weight_power==.5 and c.leaves==63 for c in delivered))

    def test_linear_leader_refinements_do_not_mutate_unused_tree_parameters(self):
        for classification in (False,True):
            with self.subTest(classification=classification),tempfile.TemporaryDirectory() as tmp:
                task,*_=load_task(make_task(Path(tmp),classification=classification))
                planner=Mock();planner.available=False
                research=Research(task,{},planner)
                original=ExperimentSpec(model="linear")
                research.queue=[];research.attempted.add(original.key)
                history=[{"status":"ok","score":.6 if classification else 1.,"config":original.to_dict()}]
                keys=[]
                for _ in range(3):
                    candidate=research.next(history)
                    self.assertIsNotNone(candidate)
                    changed={key for key,value in candidate.to_dict().items() if value!=original.to_dict()[key]}
                    self.assertLessEqual(changed,{"regularization","class_weight_power","hypothesis"})
                    self.assertNotEqual(candidate.key,original.key)
                    keys.append(candidate.key)
                self.assertEqual(len(set(keys)),3)


if __name__=='__main__':unittest.main()
