"""Закрытые схемы исследовательских преобразований; произвольный код запрещён."""
from dataclasses import asdict, dataclass, field
from contracts import fingerprint, parse_experiment


@dataclass
class AugmentSpec:
    model: str = "augmentation"
    base_config: dict = field(default_factory=dict)
    feature_schema: dict = field(default_factory=dict)
    history_schema: dict = field(default_factory=dict)
    seed: int = 42
    hypothesis: str = "Проверка новых признаков"

    def validate(self, task):
        if self.model != "augmentation" or not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**31:
            raise ValueError("Некорректная конфигурация преобразований")
        if any(not isinstance(v,dict) for v in (self.base_config,self.feature_schema,self.history_schema)):
            raise ValueError("Схемы преобразований должны быть объектами")
        if task.target in task.feature_columns or set(task.id_columns)&set(task.feature_columns):
            raise ValueError("Признаки не могут содержать target/id")
        if self.base_config.get("model") not in {"catboost", "lightgbm", "extra_trees"}:
            raise ValueError("Недопустимая базовая модель")
        parse_experiment(self.base_config, task)
        if not isinstance(self.hypothesis, str) or len(self.hypothesis) > 1200:
            raise ValueError("Некорректная гипотеза")
        forbidden = task.id_columns+[task.target]
        if self.feature_schema:
            from kaggle_features import FoldFeatureAugmenter
            FoldFeatureAugmenter(self.feature_schema, forbidden_columns=forbidden, seed=self.seed).validate_columns(
                task.feature_columns, task.categorical_columns)
        if self.history_schema:
            from historical_features import HistorySchema
            schema = HistorySchema(**self.history_schema).validate()
            if task.classification:
                raise ValueError("История числового таргета неприменима к классам")
            if set(schema.entity_columns) & set(forbidden) or schema.time_column in forbidden:
                raise ValueError("Запрещённая колонка истории")
            if not (set(schema.entity_columns) | {schema.time_column}) <= set(task.feature_columns):
                raise ValueError("История использует неизвестные признаки")
        return self

    def to_dict(self):
        return asdict(self)

    @property
    def key(self):
        value = self.to_dict()
        value.pop("hypothesis")
        value["base_config"] = dict(value["base_config"])
        value["base_config"].pop("hypothesis", None)
        return fingerprint(value)

    @classmethod
    def from_dict(cls, data, task):
        return cls(**data).validate(task)


def parse_research_spec(data, task):
    if data.get("model") == "augmentation":
        return AugmentSpec.from_dict(data, task)
    if data.get("model") == "xgboost":
        from kaggle_xgb import XGBSpec
        return XGBSpec.from_dict(data, task)
    return parse_experiment(data, task)
