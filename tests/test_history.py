import unittest

import numpy as np
import pandas as pd

from historical_features import HISTORY_COLUMNS, HistoricalFeatures, HistorySchema, suggest_history_schema


class HistoryTests(unittest.TestCase):
    def schema(self, **changes):
        args = dict(time_column="period", entity_columns=("region", "area"),
                    feature_columns=("period", "region", "area", "x"),
                    forbidden_columns=("id", "label"))
        args.update(changes)
        return HistorySchema(**args)

    def data(self):
        # Unsorted periods and duplicate DataFrame indices must not affect results.
        return pd.DataFrame(dict(period=[3, 1, 2, 2, 4], region=["a"]*5,
                                 area=[10.]*5, x=[0.]*5), index=[8, 8, 2, 1, 0])

    def test_strictly_past_labels_and_same_period_aggregation(self):
        frame = self.data()
        model = HistoricalFeatures(self.schema())
        result = model.fit_transform(frame, [1000, 10, 20, 40, 9000])
        np.testing.assert_allclose(result["__history_mean"], [70/3, np.nan, 10, 10, 1070/4], equal_nan=True)
        np.testing.assert_allclose(result["__history_last"], [30, np.nan, 10, 10, 1000], equal_nan=True)
        np.testing.assert_array_equal(result["__history_count"], [3, 0, 1, 1, 4])
        self.assertEqual(model.diagnostics_["multirow_key_period_groups"], 1)
        self.assertEqual(model.diagnostics_["different_target_key_period_groups"], 1)
        np.testing.assert_array_equal(result.index, frame.index)

    def test_future_target_changes_cannot_change_earlier_features(self):
        frame = self.data()
        first = HistoricalFeatures(self.schema()).fit_transform(frame, [1000, 10, 20, 40, 9000])
        second = HistoricalFeatures(self.schema()).fit_transform(frame, [100000, 10, 20, 40, -9e8])
        early = frame.period <= 3
        np.testing.assert_allclose(first.loc[early, HISTORY_COLUMNS], second.loc[early, HISTORY_COLUMNS], equal_nan=True)

    def test_transform_excludes_present_and_future_and_never_reads_target(self):
        frame = self.data()
        model = HistoricalFeatures(self.schema()).fit(frame, [1000, 10, 20, 40, 9000])
        query = pd.DataFrame(dict(period=[2, 3, 5], region=["a"]*3, area=[10.]*3,
                                  x=[1e100]*3, label=[-1e100]*3, id=["xx"]*3))
        result = model.transform(query)
        np.testing.assert_allclose(result["__history_last"], [10, 30, 9000])
        query.label = 1e100
        np.testing.assert_allclose(model.transform(query)[list(HISTORY_COLUMNS)],
                                   result[list(HISTORY_COLUMNS)], equal_nan=True)

    def test_unknown_missing_keys_and_invalid_time_have_no_global_prior(self):
        model = HistoricalFeatures(self.schema()).fit(self.data(), [1000, 10, 20, 40, 9000])
        query = pd.DataFrame(dict(period=[8, 8, np.nan, 1], region=["z", "a", "a", "a"],
                                  area=[10., np.nan, 10., 10.], x=[0.]*4))
        result = model.transform(query)
        self.assertTrue(result["__history_mean"].isna().all())
        self.assertTrue((result["__history_count"] == 0).all())
        self.assertEqual(model.last_diagnostics_["coverage"], 0.)
        self.assertEqual(len(model.transform(query.iloc[:0])), 0)

    def test_composite_keys_do_not_collide_on_string_separator_or_types(self):
        frame = pd.DataFrame(dict(period=[1, 1, 2, 2], region=["a_b", "a", "a_b", "a"],
                                  area=["c", "b_c", "c", "b_c"], x=[0.]*4))
        result = HistoricalFeatures(self.schema()).fit_transform(frame, [10., 90., 0., 0.])
        np.testing.assert_allclose(result["__history_mean"].iloc[2:], [10, 90])

    def test_schema_rejects_forbidden_unknown_and_invalid_options(self):
        invalid = [self.schema(entity_columns=("id",)), self.schema(entity_columns=("label",)),
                   self.schema(entity_columns=("unknown",)), self.schema(entity_columns=("period",)),
                   self.schema(entity_columns=("area", "area")), self.schema(min_count=True)]
        for schema in invalid:
            with self.assertRaises(ValueError):
                HistoricalFeatures(schema)
        with self.assertRaises(ValueError):
            HistorySchema.from_dict({"time_column": "period", "entity_columns": ["region"],
                                     "code": "exec"}, ["period", "region"])
        with self.assertRaises(ValueError):
            HistoricalFeatures(self.schema()).fit(self.data(), [1])

    def test_metadata_suggestion_uses_descriptions_and_requires_complete_roles(self):
        descriptions=["Код штата", "Класс здания", "Назначение здания", "Площадь здания", "Год постройки"]
        columns=["custom_region","custom_kind","custom_use","custom_size","custom_construction","period"]
        dd=pd.DataFrame({"feature":columns,"role":["feature"]*6,
                         "dtype":["category"]*3+["numeric"]*3,"description_ru":descriptions+["Код периода"]})
        schema=suggest_history_schema(dd,columns,"period",["id","label"])
        self.assertEqual(schema["entity_columns"],columns[:5])
        self.assertEqual(suggest_history_schema(dd.iloc[:-1],columns,"period"),{})
        self.assertEqual(suggest_history_schema(dd,columns,"period",["custom_size"]),{})


if __name__ == "__main__":
    unittest.main()
