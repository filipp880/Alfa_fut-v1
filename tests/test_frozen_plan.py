"""План выбирается поиском, а контроль не заменяется удачным confirm-кандидатом."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from contracts import ExperimentSpec
from fixtures import make_task
from history_router import HistoryRouterSpec
from proposal_models import ProposalSpec
from research import Research, cap_iterations, control_plan, search_plan, choose_confirmed_plan
from task_reader import load_task


class FrozenPlanTests(unittest.TestCase):
    def test_fixed_classification_control_cannot_be_replaced_by_llm_lookalike(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, *_ = load_task(make_task(Path(tmp),classification=True))
        cb=ExperimentSpec(class_weight_power=.5)
        lgb=ExperimentSpec(model="lightgbm",iterations=2400,leaves=63,min_child_samples=60,class_weight_power=.5)
        rows=[{"config_key":"lookalike","config":replace(cb,iterations=6000).to_dict(),"score":.9},
              {"config_key":"cb","config":cb.to_dict(),"score":.5},
              {"config_key":"lgb","config":lgb.to_dict(),"score":.5}]
        probability=np.array([[1.,0.,0.],[0.,1.,0.]])
        plan=control_plan(task,{"minority_fraction":.01},rows,np.array([0,1]),lambda r:probability)
        self.assertEqual(plan["members"],["lgb","cb"])
        self.assertEqual(plan["weights"],[.5,.5])
        self.assertAlmostEqual(plan["score"],2/3)
    def test_classification_guard_is_prioritized_without_removing_previous_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, *_ = load_task(make_task(Path(tmp),classification=True))
        planner = type("Planner",(),{"available":False})()
        queue = Research(task,{"minority_fraction":.01},planner).queue
        weighted = [c for c in queue if c.model=="lightgbm" and c.class_weight_power==.5]
        self.assertEqual(queue[0].model,"linear")
        self.assertTrue(any(c.leaves==63 and c.min_child_samples==60 for c in queue[:5]))
        self.assertTrue(any(c.leaves==31 for c in weighted))
    def test_search_plan_freezes_complete_useful_composition(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, *_ = load_task(make_task(Path(tmp)))
        y = np.zeros(4); arrays = {"a":np.array([1.,1.,-1.,-1.]),"b":np.array([-2.,-2.,2.,2.])}
        records = [{"config_key":k,"score":v,"config":{"model":m}} for k,v,m in [("a",1.,"catboost"),("b",2.,"extra_trees")]]
        plan = search_plan(task,records,y,lambda r:arrays[r["config_key"]])
        self.assertEqual(plan["members"],["a","b"])
        self.assertEqual(plan["weights"],[.75,.25])
        self.assertAlmostEqual(plan["score"],.25)

    def test_known_forest_is_control_even_if_search_prefers_another_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, *_ = load_task(make_task(Path(tmp)))
        records = [{"config_key":"cb","score":1.,"config":{"model":"catboost"}},
                   {"config_key":"et","score":2.,"config":{"model":"extra_trees"}}]
        plan = control_plan(task,{},records,np.zeros(2),lambda r:np.ones(2))
        self.assertEqual(plan["members"],["et"])
        self.assertEqual(plan["score"],2.)

    def test_caps_reach_both_router_components_and_nested_proposal(self):
        router = cap_iterations(HistoryRouterSpec(history_schema={}),100)
        self.assertEqual((router.iterations,router.fallback_iterations),(100,100))
        proposal = cap_iterations(ProposalSpec(base_config=ExperimentSpec().to_dict()),100)
        self.assertEqual(proposal.base_config["iterations"],100)

    def test_same_router_is_not_blended_to_expand_frozen_reserve_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, *_ = load_task(make_task(Path(tmp)))
        records=[{"config_key":"a","score":1.,"config":{"model":"history_router"}},
                 {"config_key":"b","score":2.,"config":{"model":"history_router"}}]
        arrays={"a":np.array([1.,-1.]),"b":np.array([-2.,2.])}
        plan=search_plan(task,records,np.zeros(2),lambda r:arrays[r["config_key"]])
        self.assertEqual(plan["members"],["a"])

    def test_oof_cannot_replace_useful_legacy_control_with_weaker_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            task,*_=load_task(make_task(Path(tmp),classification=True))
        def plan(key):return {"members":[key],"weights":[1.],"score":.8}
        oof,legacy,control=map(plan,["oof","legacy","control"])
        scores={"oof":.60,"legacy":.55,"control":.61}
        chosen,_=choose_confirmed_plan(task,oof,legacy,control,
            lambda p:scores[p["members"][0]],True)
        self.assertEqual(chosen,control)
        scores["oof"]=.62
        chosen,_=choose_confirmed_plan(task,oof,legacy,control,
            lambda p:scores[p["members"][0]],True)
        self.assertEqual(chosen,oof)

    def test_oof_requires_complete_guard_and_more_than_tolerance_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            task,*_=load_task(make_task(Path(tmp),classification=True))
        def plan(key):return {"members":[key],"weights":[1.],"score":.8}
        oof,legacy,control=map(plan,["oof","legacy","control"])
        for scores,expected in [({"oof":.9,"legacy":None,"control":None},legacy),
                                ({"oof":None,"legacy":.62,"control":.61},legacy),
                                ({"oof":.6205,"legacy":.62,"control":.61},legacy)]:
            chosen,_=choose_confirmed_plan(task,oof,legacy,control,
                lambda p:scores[p["members"][0]],True)
            self.assertEqual(chosen,expected)

    def test_non_oof_confirmation_keeps_original_fallback_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            task,*_=load_task(make_task(Path(tmp)))
        a={"members":["a"],"weights":[1.],"score":1.}
        b={"members":["b"],"weights":[1.],"score":2.}
        scores={"a":3.,"b":2.}
        chosen,_=choose_confirmed_plan(task,a,a,b,lambda p:scores[p["members"][0]],False)
        self.assertEqual(chosen,b)


if __name__=="__main__":unittest.main()
