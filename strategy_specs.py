"""Контракты и проверки сравнительного исследования."""
from dataclasses import dataclass, replace
import re
import numpy as np
import pandas as pd

from contracts import ExperimentSpec, TaskSpec
from preprocessing import FeatureBuilder


@dataclass
class AutoGluonSpec(ExperimentSpec):
    model: str = "autogluon"
    bag_folds: int = 3

    def validate(self, task: TaskSpec):
        if self.model != "autogluon" or self.bag_folds not in {0, 2, 3}:
            raise ValueError("Недопустимая конфигурация AutoGluon")
        ExperimentSpec.validate(replace(self,model="lightgbm"),task)
        if (self.transforms or self.target_transform != "identity" or self.missing_indicators
                or self.frequency_encoding or self.drop_time):
            raise ValueError("AutoGluon использует собственную предобработку без внешних признаков")
        return self


def validate_semantic_transforms(task: TaskSpec, train: pd.DataFrame, dictionary: pd.DataFrame,
                                transforms: list[dict]) -> list[dict]:
    """Проверка единиц, области определения и дублирования на search-train."""
    units = {}
    if not dictionary.empty and "feature" in dictionary:
        units = {str(row["feature"]):str(row.get("unit","")).strip().lower()
                 for row in dictionary.fillna("").to_dict("records")}
    config = replace(ExperimentSpec(),transforms=transforms).validate(task)
    builder = FeatureBuilder(task,config,{}).fit(train)
    generated = builder.transform(train)
    accepted = []
    for i,operation in enumerate(transforms):
        op,columns = operation["op"],operation["columns"]
        if op in {"sum","difference"}:
            a,b = [units.get(c,"") for c in columns]
            if a and b and a!=b:
                continue
        value = generated[f"__derived_{i}_{op}"].to_numpy(dtype=float)
        if np.isfinite(value).sum()<min(20,len(value)*.1):
            continue
        redundant = False
        for col in task.feature_columns:
            if col in task.categorical_columns:
                continue
            original = pd.to_numeric(train[col],errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(value) & np.isfinite(original)
            if (np.array_equal(np.isfinite(value),np.isfinite(original)) and finite.any()
                    and np.allclose(value[finite],original[finite],rtol=1e-7,atol=1e-9)):
                redundant = True
                break
        if not redundant:
            accepted.append(operation)
    return accepted


DEFAULT_SPACE = {
    "depth":[4,8],"learning_rate":[.025,.12],"regularization":[.5,30.],
    "leaves":[15,127],"min_child_samples":[15,120],
}


def validated_space(proposal):
    space = {k:list(v) for k,v in DEFAULT_SPACE.items()}
    if not isinstance(proposal,dict):
        return space
    for name,outer in DEFAULT_SPACE.items():
        bounds = proposal.get(name)
        if not isinstance(bounds,list) or len(bounds)!=2:
            continue
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) for x in bounds):
            continue
        lo,hi = max(outer[0],bounds[0]),min(outer[1],bounds[1])
        if name in {"depth","leaves","min_child_samples"}:
            lo,hi = int(np.ceil(lo)),int(np.floor(hi))
        if lo<hi:
            space[name]=[lo,hi]
    return space
