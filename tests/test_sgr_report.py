"""Проверки независимых метрик/артефактов SGR, без API, Docker и обучения."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from contracts import ExperimentSpec, TaskSpec
from sgr_planner import make_sgr_schema
from data_io import encode_target, profile_data
from task_reader import load_task
from validation import make_splits
from tools.report_sgr_study import (AuditError, audit_arm, audit_csv, audit_isolation, audit_reference, build_report,
    independent_csv, independent_score, metric_details, prompt_audit, select_weight,
    source_schema_strict)


def task(classification=False, classes=None, kind=None):
    return TaskSpec("task.md", "train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv",
                    "target", "classification" if classification else "regression",
                    "macro_f1" if classification else "rmse", ["id"], ["x"], [], ["id", "target"],
                    kind or ("label" if classification else "value"), classes or [])


class IndependentMetricTests(unittest.TestCase):
    def test_rmse_uses_original_scale(self):
        self.assertAlmostEqual(independent_score(task(), np.array([100., 400.]), np.array([110., 380.])),
                               np.sqrt(250.))

    def test_macro_f1_includes_missing_contract_class(self):
        spec = task(True, ["A", "B", "C"])
        truth = np.array([0, 0, 1, 1])
        p = np.eye(3)[truth]
        self.assertAlmostEqual(independent_score(spec, truth, p), 2/3)
        details = metric_details(spec, truth, p)
        self.assertEqual(details["confusion_matrix"], [[2, 0, 0], [0, 2, 0], [0, 0, 0]])
        self.assertEqual(details["per_class_f1"], [1., 1., 0.])

    def test_probability_axis_and_finite_are_enforced(self):
        spec = task(True, ["A", "B", "C"])
        for p in (np.array([[.5, .5]]), np.array([[0., 0., 0.]]),
                  np.array([[np.nan, .5, .5]]), np.array([[-.1, .5, .6]])):
            with self.subTest(p=p), self.assertRaises(AuditError):
                independent_score(spec, np.array([0]), p)

    def test_search_grid_keeps_control_below_predefined_tolerance(self):
        truth = np.array([0., 0.])
        control = np.array([1., 1.])
        candidate = np.array([.9995, .9995])
        alpha, score, grid = select_weight(task(), truth, control, candidate)
        self.assertEqual(alpha, 0.)
        self.assertEqual(score, 1.)
        self.assertEqual(len(grid), 5)

    def test_search_grid_can_select_interior_mix(self):
        truth = np.array([0., 0.])
        alpha, score, _ = select_weight(task(), truth, np.array([1., 1.]), np.array([-1., -1.]))
        self.assertEqual(alpha, .5)
        self.assertEqual(score, 0.)


class IndependentArtifactTests(unittest.TestCase):
    def test_csv_preserves_string_ids_and_contract_label_axis(self):
        spec = task(True, ["z", "a", "NA"])
        test = pd.DataFrame({"id": ["001", "00002"], "x": [0, 1]})
        p = np.array([[0., 0., 1.], [1., 0., 0.]])
        payload = independent_csv(spec, test, p)
        self.assertEqual(payload, b"id,target\n001,NA\n00002,z\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            path.write_bytes(payload)
            recorded = {"status": "ok", "sha256": hashlib.sha256(payload).hexdigest()}
            self.assertTrue(audit_csv(spec, test, p, path, recorded)["independently_rebuilt"])
            path.write_bytes(payload.replace(b"00002,z", b"00002,a"))
            with self.assertRaises(AuditError):
                audit_csv(spec, test, p, path, recorded)

    def test_probability_csv_maps_explicit_class_axis(self):
        spec = task(True, ["z", "a", "NA"], "probabilities")
        spec.output_columns = ["id", "p_NA", "p_z", "p_a"]
        spec.probability_columns = ["p_NA", "p_z", "p_a"]
        spec.probability_indices = [2, 0, 1]
        payload = independent_csv(spec, pd.DataFrame({"id": ["007"]}), np.array([[.2, .3, .5]]))
        self.assertEqual(payload, b"id,p_NA,p_z,p_a\n007,0.5,0.2,0.3\n")

    def test_only_completed_runs_can_generate_results(self):
        # Полный workspace вне checkout сохраняет проверку runs/sgr даже при read-only bind.
        with tempfile.TemporaryDirectory(prefix="report_test_") as workspace_tmp, \
                patch("tools.report_sgr_study.ROOT", Path(workspace_tmp).resolve()):
            run = Path(workspace_tmp).resolve() / "runs" / "sgr" / "fixture"
            run.mkdir(parents=True)
            (run / "summary.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(AuditError, "ещё не завершён"):
                build_report(run)

    def test_reference_is_outside_single_candidate_bind(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "candidate"
            audit = {"status": "verified", "attested_before_fit": True,
                "actual_bind_mounts": [{"source": str(case), "destination": "/task", "rw": True}],
                "network": "none", "llm_calls": 0, "image_checkout_hidden": True,
                "no_socket_or_home_mounts": True, "readonly_root": True, "cpu_total": 8,
                "memory_bytes_total": 29*2**30}
            audit_isolation(audit, case, [Path(tmp) / "reference.npz"])
            with self.assertRaises(AuditError):
                audit_isolation(audit, case, [case / "reference.npz"])
            audit["actual_bind_mounts"].append({"source": str(Path(tmp)), "destination": "/other", "rw": False})
            with self.assertRaises(AuditError):
                audit_isolation(audit, case, [])

    def test_static_strict_schema_wrapper(self):
        strict = "options['response_format']={'type':'json_schema','json_schema':{'name':'x','strict':True,'schema':response_schema}}"
        self.assertTrue(source_schema_strict(strict))
        self.assertFalse(source_schema_strict(strict.replace("True", "False")))

    def test_prompt_has_no_history_and_usage_fallback_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            spec = replace(task(), task_path=str(directory / "task.md"))
            Path(spec.task_path).write_text("Метрика RMSE", encoding="utf-8")
            payload = {"task_excerpt": "Метрика RMSE", "task": {"type": "regression", "metric": "rmse",
                "features": ["x"], "classes": []}, "profile": {"experiment_seed": 42}, "history": [],
                "remaining_seconds": 10799, "example": ExperimentSpec().to_dict()}
            schema = make_sgr_schema(spec)
            prompt = {"system": "исследователь", "payload": payload, "response_schema": schema,
                      "other_policy_context": False, "previous_reports_in_context": False}
            (directory / "prompt.json").write_text(json.dumps(prompt, ensure_ascii=False), encoding="utf-8")
            (directory / "experiments.jsonl").write_text(json.dumps({"event": "llm_error", "call": 1,
                "error_type": "APIConnectionError"}) + "\n", encoding="utf-8")
            estimate = (len(prompt["system"])+len(json.dumps(payload, ensure_ascii=False, default=str))
                        +len(json.dumps(schema, ensure_ascii=False)))//2+1400
            arm = {"policy": "sgr", "seed": 42, "valid_proposal": False, "calls": 1, "tokens": estimate}
            ref = {"task": spec, "profile": {}, "frozen_paths": [], "system_prompts": {"sgr": "исследователь"}}
            result = prompt_audit(directory, arm, ref)
            self.assertEqual(result["usage"]["estimated_after_error"], estimate)
            prompt["payload"]["history"] = [{"score": 1.}]
            (directory / "prompt.json").write_text(json.dumps(prompt, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(AuditError):
                prompt_audit(directory, arm, ref)

    def test_complete_synthetic_arm_and_tampered_frozen_count(self):
        """Полная цепочка реальных форматов файлов; predictions заданы без fit."""
        with tempfile.TemporaryDirectory(prefix="report_fixture_") as workspace_tmp, \
                patch("tools.report_sgr_study.ROOT", Path(workspace_tmp).resolve()):
            workspace = Path(workspace_tmp).resolve()
            run = workspace / "runs" / "sgr" / "fixture"
            release = workspace / "runs" / "release" / "fixture"
            run.mkdir(parents=True)
            release.mkdir(parents=True)

            def digest(path):
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()

            def dump(path, value):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

            def record(path):
                return {"sha256": digest(path), "bytes": Path(path).stat().st_size}

            def isolation(case):
                return {"status": "verified", "attested_before_fit": True,
                    "actual_bind_mounts": [{"source": str(case), "destination": "/task", "rw": True}],
                    "network": "none", "llm_calls": 0, "image_checkout_hidden": True,
                    "no_socket_or_home_mounts": True, "readonly_root": True, "cpu_total": 8,
                    "memory_bytes_total": 29*2**30}

            (release / "task.md").write_text("Регрессия. Основная метрика RMSE.", encoding="utf-8")
            train = pd.DataFrame({"id": [f"{i:03}" for i in range(20)], "period": np.repeat(np.arange(1, 6), 4),
                                  "x": np.tile(np.arange(4), 5)})
            train["target"] = train["period"]*10. + train["x"]
            test = pd.DataFrame({"id": ["00021", "00022"], "period": [6, 6], "x": [0, 1]})
            train.to_csv(release / "train.csv", index=False)
            test.to_csv(release / "test.csv", index=False)
            pd.DataFrame({"id": test["id"], "target": [0., 0.]}).to_csv(release / "sample_submission.csv", index=False)
            pd.DataFrame({"feature": ["id", "period", "x", "target"], "role": ["id", "feature", "feature", "target"]}).to_csv(
                release / "data_dictionary.csv", index=False)
            spec, train, test, _, dictionary = load_task(release / "task.md")
            y = encode_target(spec, train)
            _, groups, times = profile_data(spec, train, test, dictionary)
            search, confirm = make_splits(spec, y, groups, times)
            client = ("options['response_format']={'type':'json_schema','json_schema':{'name':'x','strict':True,'schema':response_schema}}\n"
                      "class LLMPlanner:\n    def propose(self):\n        return self.request('исследователь', {})\n")
            (release / "llm_client.py").write_text(client, encoding="utf-8")
            sgr = "class SGRPlanner:\n    def propose(self):\n        return self.request('SGR исследователь', {})\n"
            (release / "sgr_planner.py").write_text(sgr, encoding="utf-8")
            with zipfile.ZipFile(release / "source.zip", "w") as bundle:
                bundle.write(release / "llm_client.py", "llm_client.py")
                bundle.write(release / "sgr_planner.py", "sgr_planner.py")
            archive_hash = digest(release / "source.zip")
            plan = {"members": ["base"], "weights": [1.]}
            mainwork = release / ".alfa_runs" / "fixture"
            dump(mainwork / "summary.json", {"status": "completed", "refit_completed": True, "selected_plan": plan})
            dump(mainwork / "profile.json", {"time_column": "period"})
            baseline_test = np.array([12., 13.])
            for phase, split in (("search", search), ("confirm", confirm), ("refit", None)):
                work = mainwork / phase
                work.mkdir()
                dump(work / "controller_result.json", {"status": "ok", "phase": phase, "config_key": "base"})
                va = split.valid if split else np.empty(0, dtype=np.int64)
                np.savez_compressed(work / "predictions.npz", valid=y[va]+2., test=baseline_test, valid_indices=va)
            dump(release / "release_result.json", {"status": "completed", "refit_completed": True, "archive_sha256": archive_hash})
            dump(release / "release_manifest.json", {"archive_sha256": archive_hash})
            dump(release / "resource_audit.json", isolation(release))
            snapshot = run / "sources" / "regression"
            snapshot.mkdir(parents=True)
            (snapshot / "llm_client.py").write_text(client, encoding="utf-8")
            (snapshot / "sgr_planner.py").write_text(sgr, encoding="utf-8")
            (snapshot / "isolated_trial.py").write_text("# frozen test instrumentation\n", encoding="utf-8")
            metadata = {"release_case": release.relative_to(workspace).as_posix(), "archive_sha256": archive_hash,
                "plan": plan, "train_sha256": digest(release / "train.csv"), "test_sha256": digest(release / "test.csv"),
                "search_score": 2., "confirm_score": 2.,
                "instrumentation_sha256": {"isolated_trial.py": digest(snapshot / "isolated_trial.py")}}
            dump(run / "reference_regression.json", metadata)
            np.savez_compressed(run / "reference_regression.npz", search=y[search.valid]+2., confirm=y[confirm.valid]+2.,
                                full_test=baseline_test, search_indices=search.valid, confirm_indices=confirm.valid)
            ref = audit_reference(run, "regression")
            self.assertEqual(ref["control"], {"search": 2., "confirm": 2.})
            label = "regression_normal_43"
            directory = run / "prompts" / label
            directory.mkdir(parents=True)
            proposal = ExperimentSpec(model="linear", iterations=50).to_dict()
            config = {**proposal, "seed": 43}
            payload = {"task_excerpt": Path(spec.task_path).read_text(encoding="utf-8"),
                "task": {"type": spec.problem_type, "metric": spec.metric, "features": spec.feature_columns, "classes": []},
                "profile": {**ref["profile"], "experiment_seed": 43}, "history": [],
                "remaining_seconds": 10799, "example": ExperimentSpec().to_dict()}
            dump(directory / "prompt.json", {"system": "исследователь", "payload": payload, "response_schema": None,
                "other_policy_context": False, "previous_reports_in_context": False})
            events = [{"event": "llm_usage", "call": 1, "tokens": 100, "total_tokens": 100, "completion_tokens": 20},
                      {"event": "llm_hypothesis", "config": proposal}]
            (directory / "experiments.jsonl").write_text("\n".join(json.dumps(value) for value in events), encoding="utf-8")
            arm = {"label": label, "task": "regression", "policy": "normal", "seed": 43, "calls": 1,
                   "tokens": 100, "valid_proposal": True, "status": "completed", "source_reports_available": False,
                   "original_proposal": proposal, "executed_config": config, "jobs": [], "elapsed_seconds": 3.}
            case = run / "trials" / label
            case.mkdir(parents=True)
            contract = spec.to_dict()
            for name, filename in (("task_path", "task.md"), ("train_path", "train.csv"), ("test_path", "test.csv"),
                                   ("sample_path", "sample_submission.csv"), ("dictionary_path", "data_dictionary.csv")):
                contract[name] = "/task/" + filename
            for filename in ("task.md", "test.csv", "sample_submission.csv", "data_dictionary.csv"):
                (case / filename).write_bytes((release / filename).read_bytes())
            for filename in ref["sources"]:
                (case / filename).write_bytes((snapshot / filename).read_bytes())
            dump(case / "config.json", config)
            dump(case / "task_contract.json", contract)
            dump(case / "split_manifest.json", {"target_exported": False, "contract_classes_frozen": True,
                 "seed": 42, "time_column": "period"})
            (case / "splits").mkdir()
            splits = {"search_late": (search.train, search.valid), "confirm": (confirm.train, confirm.valid),
                      "full": (np.arange(len(train)), np.empty(0, dtype=np.int64))}
            for phase, (tr, va) in splits.items():
                np.savez_compressed(case / "splits" / (phase + ".npz"), train=tr, valid=va)
            immutable = {str(path.relative_to(case)).replace("\\", "/"): record(path)
                         for path in case.rglob("*") if path.is_file()}
            manifest = {"label": label, "original_train_mounted": False, "llm_calls": 0,
                        "source_files": list(ref["sources"]), "immutable_files": immutable,
                        "original_input_hashes": {filename: record(release / filename) for filename in
                            ("task.md", "train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv")}}
            dump(case / "manifest.json", manifest)
            manifest_hash = digest(case / "manifest.json")
            frozen = {"config": config, "reference": metadata, "source_manifest_sha256": manifest_hash,
                      "best_iterations": 4, "weight": 1., "search_score": 1., "search_blend_score": 1.}
            arm["frozen"] = frozen
            final = np.array([10., 11.])
            for phase, (tr, va) in splits.items():
                phase_dir = case / "phases" / phase
                phase_dir.mkdir(parents=True)
                np.savez_compressed(phase_dir / "split.npz", train=tr, valid=va)
                np.savez_compressed(phase_dir / "predictions.npz", valid=y[va]+1., test=final, valid_indices=va)
                count = None if phase == "search_late" else 4
                allowed = np.union1d(tr, va) if phase == "search_late" else tr
                mask = {"allowed_label_indices_sha256": hashlib.sha256(np.asarray(allowed, dtype="<i8").tobytes()).hexdigest(),
                        "allowed_label_rows": len(allowed), "masked_label_rows": len(train)-len(allowed),
                        "original_train_mounted": False, "original_train_sha256": metadata["train_sha256"],
                        "raw_feature_values_preserved": True, "validation_labels_available": phase == "search_late",
                        "label_blinded": phase == "confirm"}
                dump(phase_dir / "masking_audit.json", mask)
                attest = {**isolation(case), "manifest_sha256": manifest_hash, "masking_audit": mask}
                dump(phase_dir / "runtime_isolation_attestation.json", attest)
                dump(phase_dir / "host_request.json", {"phase": phase, "fixed_iterations": count, "seed": None,
                                                      "trusted_original_mounted": False})
                dump(phase_dir / "frozen_request.json", {"fixed_iterations": count})
                profile, _, _ = profile_data(spec, train.iloc[tr], test, dictionary)
                profile["time_column"] = "period"
                dump(phase_dir / "request.json", {"task": contract, "config": config, "profile": profile,
                    "split_path": f"/task/phases/{phase}/split.npz", "threads": 8, "refit": phase != "search_late",
                    "fixed_iterations": count, "job_options": {}})
                prediction_hash = digest(phase_dir / "predictions.npz")
                dump(phase_dir / "summary.json", {"status": "completed", "config": config,
                    "source_manifest_sha256": manifest_hash, "label_blinded": phase == "confirm", "score": None,
                    "metrics": {}, "prediction_sha256": prediction_hash, "best_iterations": 4, "fixed_iterations": count})
                dump(phase_dir / "result.json", {"score": None, "metrics": {}})
                arm["jobs"].append({"phase": phase, "directory": phase_dir.relative_to(run).as_posix(),
                                    "label_blinded": phase == "confirm", "prediction_sha256": prediction_hash, "seconds": 1.})
                if phase == "search_late":
                    dump(directory / "frozen_before_confirm.json", frozen)
            csv = b"id,target\n00021,10.0\n00022,11.0\n"
            (directory / "submission.csv").write_bytes(csv)
            arm.update(selected_weight=1., standalone_confirm=1., frozen_blend_confirm=1., control_confirm=2.,
                       final_confirm=1., submission_audit={"status": "ok", "sha256": hashlib.sha256(csv).hexdigest()})
            result = audit_arm(run, arm, ref)
            self.assertTrue(result["ml_completed"])
            self.assertEqual(result["confirm_effective"], 1.)
            dump(case / "phases" / "confirm" / "host_request.json", {"phase": "confirm", "fixed_iterations": 5,
                                                                      "trusted_original_mounted": False})
            with self.assertRaisesRegex(AuditError, "count"):
                audit_arm(run, arm, ref)


if __name__ == "__main__":
    unittest.main()
