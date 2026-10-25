import unittest

import numpy as np
import pandas as pd

from kaggle_features import FoldFeatureAugmenter, regression_feature_schema


class KaggleFeatureTests(unittest.TestCase):
    def data(self):
        n = 80
        x = np.arange(n, dtype=float)
        rating = x * 2.
        rating[::4] = np.nan
        return pd.DataFrame({"x": x, "z": -x, "rating": rating,
                             "category": ["a", "b"] * (n // 2),
                             "target": x * 100., "id": np.arange(n)})

    def schema(self):
        return {"pca_groups": [{"name": "weather", "columns": ["x", "z"], "components": 2}],
                "impute_columns": ["rating"], "imputer_estimators": 8}

    def test_train_only_and_forbidden_predictors(self):
        frame = self.data()
        model = FoldFeatureAugmenter(self.schema(), ["target", "id"]).fit(frame)
        encoded = model.transform(frame)
        self.assertNotIn("target", encoded)
        self.assertNotIn("id", encoded)
        info = model.diagnostics_["imputation"]["rating"]
        self.assertEqual(info["model_rows"], 60)
        self.assertFalse({"rating", "target", "id"} & set(info["predictors"]))
        known = frame.rating.notna()
        np.testing.assert_array_equal(encoded.rating[known], frame.rating[known])
        self.assertTrue(np.isfinite(encoded.rating).all())
        self.assertEqual(encoded.rating__imputed_missing.sum(), 20)
        changed = frame.copy()
        changed["target"] = 1e30
        changed["id"] = "arbitrary"
        np.testing.assert_array_equal(model.transform(changed).rating, encoded.rating)

    def test_validation_does_not_refit_and_handles_unknown_empty(self):
        frame = self.data()
        model = FoldFeatureAugmenter(self.schema(), ["target", "id"]).fit(frame)
        before = model.pca_[0][3].mean_.copy()
        validation = frame.iloc[:3].copy()
        validation["x"] = 1e6
        validation["category"] = "unknown"
        validation["rating"] = np.nan
        values = model.transform(validation)
        np.testing.assert_equal(before, model.pca_[0][3].mean_)
        self.assertTrue(np.isfinite(values.rating).all())
        self.assertEqual(len(model.transform(validation.iloc[:0])), 0)
        self.assertEqual(values.index.tolist(), validation.index.tolist())

    def test_pca_is_independent_of_imputation_choice(self):
        frame = self.data()
        one = FoldFeatureAugmenter(self.schema(), ["target", "id"]).fit_transform(frame)
        schema = self.schema()
        schema["impute_columns"] = []
        two = FoldFeatureAugmenter(schema, ["target", "id"]).fit_transform(frame)
        np.testing.assert_array_equal(one["__pca_weather_1"], two["__pca_weather_1"])

    def test_forbidden_unknown_and_resource_bounds_rejected(self):
        bad = [{"impute_columns": ["target"]}, {"impute_columns": ["unknown"]},
               {"impute_columns": ["category"]}, {"imputer_estimators": 10000},
               {"pca_groups": [{"name": "bad/path", "columns": ["x", "z"], "components": 2}]},
               {"arbitrary_code": "run"}]
        for schema in bad:
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                FoldFeatureAugmenter(schema, ["target", "id"]).fit(self.data())

    def test_all_missing_column_uses_train_fallback(self):
        frame = self.data()
        frame["rating"] = np.nan
        model = FoldFeatureAugmenter({"impute_columns": ["rating"]}, ["target", "id"])
        values = model.fit_transform(frame)
        self.assertTrue((values.rating == 0).all())
        self.assertEqual(model.diagnostics_["imputation"]["rating"]["status"], "median")

    def test_dictionary_schema_does_not_include_target_or_id(self):
        dictionary = pd.DataFrame([
            {"feature": "a", "role": "feature", "dtype": "numeric", "description_ru": "Максимальная температура"},
            {"feature": "b", "role": "feature", "dtype": "numeric", "description_ru": "Сумма превышений температуры"},
            {"feature": "r", "role": "feature", "dtype": "numeric", "description_ru": "Рейтинг энергоэффективности"},
            {"feature": "target", "role": "target", "dtype": "numeric", "description_ru": "Максимальная температура"}])
        schema = regression_feature_schema(dictionary, ["a", "b", "r", "target"])
        self.assertEqual(schema["impute_columns"], ["r"])
        self.assertEqual(schema["pca_groups"][0]["columns"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
