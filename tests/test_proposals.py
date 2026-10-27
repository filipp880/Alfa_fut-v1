"""Проверки новых операторов и изоляции подтверждающего таргета."""
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
import pandas as pd
from contracts import ExperimentSpec
from data_io import encode_target,profile_data
from fixtures import make_task
from proposal_models import (NumericPolicy,ProposalSpec,conflict_weights,macro_f1_metric,
    prior_bias,fit_sigmoid_calibration,sigmoid_calibration)
from runtime import Budget,Runner
from task_reader import load_task
from validation import make_splits


class ProposalTests(unittest.TestCase):
    def test_numeric_policies_fit_only_and_keep_meaningful_zero(self):
        train=pd.DataFrame({"year":[1900.,2000.,np.nan,0.],"days":[0.,4.,np.nan,2.],"g":["a"]*4})
        policy=NumericPolicy("sentinel",["year"]).fit(train)
        transformed=policy.transform(train)
        self.assertTrue(np.isnan(transformed.year.iloc[3]))
        self.assertEqual(transformed.days.iloc[0],0)
        self.assertEqual(train.year.iloc[3],0)
        median=NumericPolicy("median",["year"]).fit(train)
        before=dict(median.medians)
        query=pd.DataFrame({"year":[np.nan,1e10],"days":[0,0],"g":["new","a"]})
        self.assertEqual(median.transform(query).year.iloc[0],1900.)
        self.assertEqual(before,median.medians)
        pattern=NumericPolicy("missing_pattern",["year","days"]).fit(train).transform(train)
        np.testing.assert_array_equal(pattern["__row_missing_count"],[0,0,2,0])
        self.assertTrue(np.isnan(pattern.year.iloc[2]))

    def test_conditional_fill_unknown_groups_and_small_groups_fall_back(self):
        train=pd.DataFrame({"x":[1.]*25+[9.]*25+[np.nan],"g":["a"]*25+["b"]*25+["tiny"]})
        model=NumericPolicy("conditional",["x"],["g"]).fit(train)
        out=model.transform(pd.DataFrame({"x":[np.nan]*3,"g":["a","b","new"]}))
        np.testing.assert_allclose(out.x,[1,9,5])
        self.assertEqual(len(model.transform(train.iloc[:0])),0)

    def test_conflict_weights_use_only_exact_x_and_keep_positive_weights(self):
        frame=pd.DataFrame({"x":[1,1,1,2,2,3]})
        weights,info=conflict_weights(frame,np.array([0,0,1,1,1,0]))
        self.assertLess(weights[2],weights[0])
        self.assertGreater(weights[3],weights[0])
        self.assertTrue((weights>0).all())
        self.assertEqual(info["conflicting_rows"],3)

    def test_probability_rules_are_finite_and_identity_bias_preserves_predictions(self):
        p=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.2,.7]]*10)
        y=np.array([0,1,2]*10)
        np.testing.assert_allclose(prior_bias(p,[.8,.15,.05],0),p)
        parameters=fit_sigmoid_calibration(p,y)
        transformed=sigmoid_calibration(np.array([[1,0,0],[0,1,0]],dtype=float),parameters)
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_allclose(transformed.sum(axis=1),1.)
        self.assertAlmostEqual(macro_f1_metric(y,p)[1],1.)

    def test_preflight_rejects_forbidden_columns_and_wrong_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            task,*_=load_task(make_task(Path(tmp)))
            base=ExperimentSpec().to_dict()
            for recipe in [ProposalSpec(base_config=base,columns=[task.target]),
                           ProposalSpec(base_config=base,columns=task.id_columns),
                           ProposalSpec(base_config=base,variant="exec"),
                           ProposalSpec(base_config=base,variant="lgb_f1")]:
                with self.assertRaises((ValueError,TypeError)):
                    recipe.validate(task)

    def test_confirm_predictions_do_not_use_validation_labels_for_new_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            task,train,test,_,dd=load_task(make_task(root,temporal=True))
            y=encode_target(task,train);profile,groups,times=profile_data(task,train,test,dd)
            confirm=make_splits(task,y,groups,times)[1]
            np.savez_compressed(root/"confirm.npz",train=confirm.train,valid=confirm.valid)
            runner=Runner(root/"jobs",task,profile,Budget(time.monotonic(),120,1,2048),worker_module="kaggle_worker")
            for variant in ("linear_residual","log_smear","linear_tree","huber"):
                recipe=ProposalSpec(variant=variant,base_config=ExperimentSpec(model="lightgbm",iterations=50,min_child_samples=10).to_dict())
                train.to_csv(task.train_path,index=False)
                first=runner.run(recipe,root/"confirm.npz",variant+"_first",20,refit=True,fixed_iterations=10)
                self.assertEqual(first["status"],"ok",first)
                changed=train.copy();changed.loc[confirm.valid,task.target]+=1000
                changed.to_csv(task.train_path,index=False)
                second=runner.run(recipe,root/"confirm.npz",variant+"_changed",20,refit=True,fixed_iterations=10)
                self.assertEqual(second["status"],"ok",second)
                with np.load(Path(first["directory"])/"predictions.npz") as a,np.load(Path(second["directory"])/"predictions.npz") as b:
                    np.testing.assert_array_equal(a["valid"],b["valid"])
                    np.testing.assert_array_equal(a["test"],b["test"])
                self.assertNotEqual(first["score"],second["score"])

    def test_classification_f1_and_conflict_workers_freeze_confirm_iterations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            task,train,test,_,dd=load_task(make_task(root,classification=True))
            y=encode_target(task,train);profile,groups,times=profile_data(task,train,test,dd)
            confirm=make_splits(task,y,groups,times)[1]
            np.savez_compressed(root/"confirm.npz",train=confirm.train,valid=confirm.valid)
            runner=Runner(root/"jobs",task,profile,Budget(time.monotonic(),90,1,2048),worker_module="kaggle_worker")
            for variant in ("lgb_f1","conflict_weight"):
                recipe=ProposalSpec(variant=variant,base_config=ExperimentSpec(model="lightgbm",iterations=50,min_child_samples=10,class_weight_power=.5).to_dict())
                train.to_csv(task.train_path,index=False)
                first=runner.run(recipe,root/"confirm.npz",variant+"_first",20,refit=True,fixed_iterations=15)
                changed=train.copy()
                changed.loc[confirm.valid,task.target]=[task.classes[(int(v)+1)%len(task.classes)] for v in y[confirm.valid]]
                changed.to_csv(task.train_path,index=False)
                second=runner.run(recipe,root/"confirm.npz",variant+"_changed",20,refit=True,fixed_iterations=15)
                for result in (first,second):
                    self.assertEqual(result["status"],"ok",result)
                    self.assertFalse(result["evaluation_used_for_fit"])
                    self.assertEqual(result["best_iterations"],15)
                with np.load(Path(first["directory"])/"predictions.npz") as a,np.load(Path(second["directory"])/"predictions.npz") as b:
                    np.testing.assert_array_equal(a["valid"],b["valid"])
                    np.testing.assert_array_equal(a["test"],b["test"])


if __name__=="__main__":
    unittest.main()
