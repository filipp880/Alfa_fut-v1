"""Заранее заданные A/B-рецепты, общие правила схем по словарю задачи."""
from dataclasses import replace
import re
from contracts import ExperimentSpec
from historical_features import suggest_history_schema
from kaggle_specs import AugmentSpec,parse_research_spec
from novel_models import NovelSpec
from proposal_models import ProposalSpec


def parse(data,task):
    return ProposalSpec.from_dict(data,task) if data.get("model")=="proposal" else parse_research_spec(data,task)


def catalog(task,dictionary,profile):
    controls={}
    if task.classification:
        controls["lgb"]=ExperimentSpec(model="lightgbm",iterations=2400,leaves=63,min_child_samples=60,class_weight_power=.5)
        controls["cb"]=ExperimentSpec(class_weight_power=.5)
        specs={"G4":ProposalSpec(variant="lgb_f1",base_config=controls["lgb"].to_dict(),hypothesis="GPT C1: остановка по Macro F1"),
               "Q6":ProposalSpec(variant="conflict_weight",base_config=controls["lgb"].to_dict(),hypothesis="Qwen: понизить вес конфликтующих одинаковых X")}
        metadata={"G4":{"source":"GPT","proposal":"LightGBM early stopping по Macro F1","paired":"lgb"},
                  "G5":{"source":"GPT","proposal":"Один коэффициент class prior bias по внутренним OOF","postprocess":"prior_bias","paired":"baseline"},
                  "Q6":{"source":"Qwen","proposal":"Веса конфликтующих меток одинаковых X","paired":"lgb"},
                  "Q7":{"source":"Qwen","proposal":"Поклассовая sigmoid-калибровка по внутренним OOF","postprocess":"sigmoid","paired":"baseline"}}
        baseline={"members":["lgb","cb"],"weights":[.5,.5]}
        return controls,specs,metadata,baseline
    controls["forest"]=NovelSpec(forest_depth=24,min_leaf=4,max_features=1.)
    controls["cb"]=ExperimentSpec(depth=8,iterations=2400,learning_rate=.045)
    controls["lgb"]=ExperimentSpec(model="lightgbm",iterations=2400)
    rows=dictionary.to_dict("records")
    numeric=[c for c in task.feature_columns if c not in task.categorical_columns]
    missing=[c for c in numeric if c in profile.get("missing_train",{})]
    calendar=[str(r["feature"]) for r in rows if r.get("feature") in numeric
              and re.search(r"год постройки|calendar year|year.*built",str(r.get("description_ru","")),re.I)]
    attributes=[str(r["feature"]) for r in rows if r.get("feature") in missing
                and re.search(r"год постройки|рейтинг|year.*built|rating",str(r.get("description_ru","")),re.I)]
    groups=list(task.categorical_columns)[:3]
    cb,lgb=controls["cb"].to_dict(),controls["lgb"].to_dict()
    specs={"G1":ProposalSpec(variant="sentinel",base_config=cb,columns=calendar,hypothesis="GPT B/P1: ноль календарного года как специальное значение"),
           "G2":ProposalSpec(variant="median",base_config=cb,columns=missing,hypothesis="GPT B/P2: медиана и индикаторы числовых пропусков"),
           "G3":ProposalSpec(variant="conditional",base_config=cb,columns=attributes,groups=groups,hypothesis="GPT B/P3: условная медиана атрибутов"),
           "Q1":ProposalSpec(variant="linear_residual",base_config=lgb,hypothesis="Qwen: Ridge и LightGBM по остаткам"),
           "Q2":ProposalSpec(variant="linear_tree",base_config=lgb,hypothesis="Qwen: линейные листья LightGBM"),
           "Q3":ProposalSpec(variant="log_smear",base_config=lgb,hypothesis="Qwen: log1p и коррекция смещения smearing"),
           "Q4":ProposalSpec(variant="huber",base_config=lgb,hypothesis="Qwen: Huber loss с оценкой RMSE"),
           "Q5":ProposalSpec(variant="missing_pattern",base_config=cb,columns=numeric,hypothesis="Qwen: паттерны пропусков и нулей")}
    metadata={"G1":{"source":"GPT","proposal":"Семантический zero→NaN + provenance indicators","paired":"cb"},
              "G2":{"source":"GPT","proposal":"Глобальная медиана + индикаторы у CatBoost","paired":"cb"},
              "G3":{"source":"GPT","proposal":"Условная медиана по статическим категориям","paired":"cb"},
              "G6":{"source":"GPT","proposal":"История с ExtraTrees-резервом для неизвестных","paired":"baseline","postprocess":"history_gate"},
              "Q1":{"source":"Qwen","proposal":"Ridge + LightGBM по остаткам","paired":"lgb"},
              "Q2":{"source":"Qwen","proposal":"LightGBM с линейными листьями","paired":"lgb"},
              "Q3":{"source":"Qwen","proposal":"log1p + train-only smearing correction","paired":"lgb"},
              "Q4":{"source":"Qwen","proposal":"Huber loss LightGBM","paired":"lgb"},
              "Q5":{"source":"Qwen","proposal":"Паттерны пропусков и нулей строки","paired":"cb"}}
    history_schema=suggest_history_schema(dictionary,task.feature_columns,profile.get("time_column"),task.id_columns+[task.target])
    if history_schema:
        specs["history_component"]=AugmentSpec(base_config=cb,history_schema=history_schema,hypothesis="Компонент G6: безопасная история")
    baseline={"members":["forest"],"weights":[1.]}
    return controls,specs,metadata,baseline
