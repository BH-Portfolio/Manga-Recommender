import pandas as pd
from unittest.mock import patch
from generate_sales import SalesDataGenerator


# Fake metadata DataFrame
@pytest.fixture
def fake_metadata():
    return pd.DataFrame({
        "manga_id": [1],
        "title": ["Berserk"],
        "members": [100000],
        "score": [9.5],
        "status": ["Publishing"],
        "volumes": [2],
        "published_from": ["2020-01-01"]
    })


@patch("generate_sales.pd.read_csv")
def test_init(mock_read_csv, fake_metadata):
    mock_read_csv.return_value = fake_metadata

    generator = SalesDataGenerator()

    assert len(generator.df) == 1
    assert "published_from" in generator.df.columns
    assert pd.api.types.is_datetime64_any_dtype(generator.df["published_from"])


@patch("generate_sales.pd.read_csv")
def test_generate_sales_trajectory(mock_read_csv, fake_metadata):
    mock_read_csv.return_value = fake_metadata

    generator = SalesDataGenerator()

    manga = generator.df.iloc[0]

    sales = generator.generate_sales_trajectory(manga)

    # 2 volumes × 12 weeks
    assert len(sales) == 24

    first = sales[0]

    assert first["manga_id"] == 1
    assert first["title"] == "Berserk"
    assert first["volume"] == 1
    assert first["week_number"] == 1
    assert first["sales_count"] >= 100


@patch("generate_sales.pd.read_csv")
def test_missing_publish_date(mock_read_csv, fake_metadata):
    fake_metadata.loc[0, "published_from"] = None

    mock_read_csv.return_value = fake_metadata

    generator = SalesDataGenerator()

    manga = generator.df.iloc[0]

    assert generator.generate_sales_trajectory(manga) == []


@patch("generate_sales.pd.read_csv")
def test_generate_all_sales(mock_read_csv, fake_metadata):
    mock_read_csv.return_value = fake_metadata

    generator = SalesDataGenerator()

    df = generator.generate_all_sales()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 24
    assert "sales_count" in df.columns
