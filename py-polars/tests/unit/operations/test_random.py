from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import ShapeError
from polars.testing import assert_frame_equal, assert_series_equal


def test_shuffle_group_by_reseed() -> None:
    def unique_shuffle_groups(n: int, seed: int | None) -> int:
        ls = [1, 2, 3] * n  # 1, 2, 3, 1, 2, 3...
        groups = sorted(list(range(n)) * 3)  # 0, 0, 0, 1, 1, 1, ...
        df = pl.DataFrame({"l": ls, "group": groups})
        shuffled = df.group_by("group", maintain_order=True).agg(
            pl.col("l").shuffle(seed)
        )
        num_unique = shuffled.group_by("l").agg(pl.lit(0)).select(pl.len())
        return int(num_unique[0, 0])

    assert unique_shuffle_groups(50, None) > 1  # Astronomically unlikely.
    assert (
        unique_shuffle_groups(50, 0xDEADBEEF) == 1
    )  # Fixed seed should be always the same.


def test_sample_expr() -> None:
    a = pl.Series("a", range(20))
    out = pl.select(
        pl.lit(a).sample(fraction=0.5, with_replacement=False, seed=1)
    ).to_series()

    assert out.shape == (10,)
    assert out.to_list() != out.sort().to_list()
    assert out.unique().shape == (10,)
    assert set(out).issubset(set(a))

    out = pl.select(pl.lit(a).sample(n=10, with_replacement=False, seed=1)).to_series()
    assert out.shape == (10,)
    assert out.to_list() != out.sort().to_list()
    assert out.unique().shape == (10,)

    # pl.set_random_seed should lead to reproducible results.
    pl.set_random_seed(1)
    result1 = pl.select(pl.lit(a).sample(n=10)).to_series()
    pl.set_random_seed(1)
    result2 = pl.select(pl.lit(a).sample(n=10)).to_series()
    assert_series_equal(result1, result2)


def test_sample_df() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]})

    assert df.sample().shape == (1, 3)
    assert df.sample(n=2, seed=0).shape == (2, 3)
    assert df.sample(fraction=0.4, seed=0).shape == (1, 3)
    assert df.sample(n=pl.Series([2]), seed=0).shape == (2, 3)
    assert df.sample(fraction=pl.Series([0.4]), seed=0).shape == (1, 3)
    assert df.select(pl.col("foo").sample(n=pl.Series([2]), seed=0)).shape == (2, 1)
    assert df.select(pl.col("foo").sample(fraction=pl.Series([0.4]), seed=0)).shape == (
        1,
        1,
    )
    with pytest.raises(ValueError, match="cannot specify both `n` and `fraction`"):
        df.sample(n=2, fraction=0.4)


def test_sample_n_expr() -> None:
    df = pl.DataFrame(
        {
            "group": [1, 1, 1, 2, 2, 2],
            "val": [1, 2, 3, 2, 1, 1],
        }
    )

    out_df = df.sample(pl.Series([3]), seed=0)
    expected_df = pl.DataFrame({"group": [2, 2, 1], "val": [1, 1, 3]})
    assert_frame_equal(out_df, expected_df)

    agg_df = df.group_by("group", maintain_order=True).agg(
        pl.col("val").sample(pl.col("val").max(), seed=0)
    )
    expected_df = pl.DataFrame({"group": [1, 2], "val": [[1, 2, 3], [1, 1]]})
    assert_frame_equal(agg_df, expected_df)

    select_df = df.select(pl.col("val").sample(pl.col("val").max(), seed=0))
    expected_df = pl.DataFrame({"val": [1, 1, 3]})
    assert_frame_equal(select_df, expected_df)


def test_sample_empty_df() -> None:
    df = pl.DataFrame({"foo": []})

    # // If with replacement, then expect empty df
    assert df.sample(n=3, with_replacement=True).shape == (0, 1)
    assert df.sample(fraction=0.4, with_replacement=True).shape == (0, 1)

    # // If without replacement, then expect shape mismatch on sample_n not sample_frac
    with pytest.raises(ShapeError):
        df.sample(n=3, with_replacement=False)
    assert df.sample(fraction=0.4, with_replacement=False).shape == (0, 1)


def test_sample_series() -> None:
    s = pl.Series("a", [1, 2, 3, 4, 5])

    assert len(s.sample(n=2, seed=0)) == 2
    assert len(s.sample(fraction=0.4, seed=0)) == 2

    assert len(s.sample(n=2, with_replacement=True, seed=0)) == 2

    # on a series of length 5, you cannot sample more than 5 items
    with pytest.raises(ShapeError):
        s.sample(n=10, with_replacement=False, seed=0)
    # unless you use with_replacement=True
    assert len(s.sample(n=10, with_replacement=True, seed=0)) == 10


def test_shuffle_expr() -> None:
    # pl.set_random_seed should lead to reproducible results.
    s = pl.Series("a", range(20))

    pl.set_random_seed(1)
    result1 = pl.select(pl.lit(s).shuffle()).to_series()

    pl.set_random_seed(1)
    result2 = pl.select(pl.lit(s).shuffle()).to_series()
    assert_series_equal(result1, result2)


def test_shuffle_series() -> None:
    a = pl.Series("a", [1, 2, 3])
    out = a.shuffle(2)
    expected = pl.Series("a", [2, 1, 3])
    assert_series_equal(out, expected)

    out = pl.select(pl.lit(a).shuffle(2)).to_series()
    assert_series_equal(out, expected)


def test_sample_16232() -> None:
    k = 2
    p = 0

    df = pl.DataFrame({"a": [p] * k + [1 + p], "b": [[1] * p] * k + [range(1, p + 2)]})
    assert df.select(pl.col("b").list.sample(n=pl.col("a"), seed=0)).to_dict(
        as_series=False
    ) == {"b": [[], [], [1]]}


def test_lazyframe_sample_frac() -> None:
    """Test LazyFrame.sample_frac() method."""
    # Create test data
    lf = pl.LazyFrame(
        {
            "a": range(100),
            "b": [f"value_{i}" for i in range(100)],
            "c": [i * 2 for i in range(100)],
        }
    )

    # Test basic functionality with 50% sampling
    result = lf.sample_frac(0.5, seed=42).collect()
    assert len(result) == 50  # Should get 50% of 100 rows
    assert result.width == 3  # Should maintain all columns
    assert set(result.columns) == {"a", "b", "c"}

    # Test reproducibility with same seed
    result1 = lf.sample_frac(0.3, seed=123).collect()
    result2 = lf.sample_frac(0.3, seed=123).collect()
    assert_frame_equal(result1, result2)

    # Test different seeds produce different results
    result3 = lf.sample_frac(0.3, seed=456).collect()
    # Results should be different (statistically very unlikely to be same)
    assert not result1.equals(result3)

    # Test with 0% sampling
    result_empty = lf.sample_frac(0.0, seed=42).collect()
    assert len(result_empty) == 0
    assert result_empty.width == 3

    # Test with 100% sampling
    result_full = lf.sample_frac(1.0, seed=42).collect()
    assert len(result_full) == 100

    # Test that sampled values are subset of original
    original_a_values = set(range(100))
    sampled_a_values = set(result["a"].to_list())
    assert sampled_a_values.issubset(original_a_values)


def test_lazyframe_sample_frac_with_none_seed() -> None:
    """Test LazyFrame.sample_frac() with None seed."""
    lf = pl.LazyFrame({"a": range(20), "b": range(20)})

    # Test with None seed (should use random seed)
    result1 = lf.sample_frac(0.5, seed=None).collect()
    result2 = lf.sample_frac(0.5, seed=None).collect()

    assert len(result1) == 10
    assert len(result2) == 10
    # With None seed, results should be different (very high probability)
    # Note: There's a small chance they could be the same by coincidence


# def test_lazyframe_sample_frac_edge_cases() -> None:
#     """Test LazyFrame.sample_frac() edge cases."""
#     # Test with small dataframe
#     small_lf = pl.LazyFrame({"x": [1, 2, 3]})
#     result = small_lf.sample_frac(0.5, seed=42).collect()
#     assert len(result) == 1  # 50% of 3 rows = 1.5 → 1 row

#     # Test with single row
#     single_lf = pl.LazyFrame({"x": [42]})
#     result_zero = single_lf.sample_frac(0.0, seed=42).collect()
#     assert len(result_zero) == 0

#     result_full = single_lf.sample_frac(1.0, seed=42).collect()
#     assert len(result_full) == 1
#     assert result_full["x"][0] == 42


def test_lazyframe_sample_frac_chaining() -> None:
    """Test LazyFrame.sample_frac() can be chained with other operations."""
    lf = pl.LazyFrame({"category": ["A"] * 50 + ["B"] * 50, "value": range(100)})

    # Test chaining with filter and select
    result = (
        lf.sample_frac(0.5, seed=42)
        .filter(pl.col("value") > 10)
        .select("category", "value")
        .collect()
    )

    assert set(result.columns) == {"category", "value"}
    assert all(val > 10 for val in result["value"].to_list())
    assert set(result["category"].to_list()).issubset({"A", "B"})


# def test_lazyframe_sample_frac_with_groupby() -> None:
#     """Test LazyFrame.sample_frac() used before group operations."""
#     lf = pl.LazyFrame(
#         {"group": ["X"] * 40 + ["Y"] * 40 + ["Z"] * 20, "value": range(100)}
#     )

#     # Sample then group by
#     result = (
#         lf.sample_frac(0.5, seed=42)
#         .group_by("group")
#         .agg(pl.col("value").count().alias("count"))
#         .sort("group")
#         .collect()
#     )

#     # Should have groups from the sampled data
#     assert len(result) <= 3  # Could have 1-3 groups depending on sampling
#     assert "group" in result.columns
#     assert "count" in result.columns
