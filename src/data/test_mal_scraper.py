import pytest
from unittest.mock import patch, MagicMock
from mal_scraper import MALScraper

@pytest.fixture
def scraper():
    return MALScraper()

# get_top_manga Tests

@patch("mal_scraper.requests.get")
@patch("mal_scraper.time.sleep")
def test_get_top_manga_success(mock_sleep, mock_get, scraper):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = { 
        "data": [{"title": "Berserk"}, {"title": "One Piece"}]
    }

    mock_get.return_value = mock_response

    result = scraper.get_top_manga(page=2, limit=10)

    assert result == [{"title": "Berserk"}, {"title": "One Piece"}]
    mock_get.assert_called_once_with(
        "https://api.jikan.moe/v4/top/manga",
        params={"page": 2, "limit": 10}
    )

    mock_sleep.assert_called_once_with(scraper.delay)


@patch("mal_scraper.requests.get")
@patch("mal_scraper.time.sleep")
def test_get_top_manga_failure(mock_sleep, mock_get, scraper, capsys):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    mock_get.return_value = mock_response

    result = scraper.get_top_manga()

    assert result == []

    captured = capsys.readouterr()
    assert "Error 500" in captured.out

    mock_sleep.assert_called_once_with(scraper.delay)


@patch("mal_scraper.requests.get")
@patch("mal_scraper.time.sleep")
def test_missing_data_key(mock_sleep, mock_get, scraper):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}

    mock_get.return_value = mock_response

    with pytest.raises(KeyError):
        scraper.get_top_manga()

# get_manga_details Tests

@patch("mal_scraper.requests.get")
@patch("mal_scraper.time.sleep")
def test_get_manga_details_success(mock_sleep, mock_get, scraper):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {"title": "Berserk", "mal_id": 1}
    }

    mock_get.return_value = mock_response

    result = scraper.get_manga_details(1)

    assert result == {"title": "Berserk", "mal_id": 1}
    mock_get.assert_called_once_with(
        "https://api.jikan.moe/v4/manga/1/full"
    )
    mock_sleep.assert_called_once_with(scraper.delay)


@patch("mal_scraper.requests.get")
@patch("mal_scraper.time.sleep")
def test_get_manga_details_failure(mock_sleep, mock_get, scraper, capsys):
    mock_response = MagicMock()
    mock_response.status_code = 404

    mock_get.return_value = mock_response

    result = scraper.get_manga_details(999)

    assert result is None

    captured = capsys.readouterr()
    assert "Error fetching manga 999" in captured.out


@patch("mal_scraper.requests.get")
@patch("mal_scraper.time.sleep")
def test_get_manga_details_missing_data(mock_sleep, mock_get, scraper):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}

    mock_get.return_value = mock_response

    with pytest.raises(KeyError):
        scraper.get_manga_details(1)

# search_manga_by_year Tests

@patch("mal_scraper.requests.get")
@patch("mal_scraper.time.sleep")
def test_search_manga_by_year_success(mock_sleep, mock_get, scraper):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"title": "Naruto"}, {"title": "Bleach"}]
    }

    mock_get.return_value = mock_response

    result = scraper.search_manga_by_year(2020, page=2, limit=5)

    assert result == [{"title": "Naruto"}, {"title": "Bleach"}]

    mock_get.assert_called_once_with(
        "https://api.jikan.moe/v4/manga",
        params={
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'order_by': 'members',
            'sort': 'desc',
            'page': 2,
            'limit': 5
        }
    )

    mock_sleep.assert_called_once_with(scraper.delay)


@patch("mal_scraper.requests.get")
@patch("mal_scraper.time.sleep")
def test_search_manga_by_year_failure(mock_sleep, mock_get, scraper):
    mock_response = MagicMock()
    mock_response.status_code = 500

    mock_get.return_value = mock_response

    result = scraper.search_manga_by_year(2020)

    assert result == []


@patch("mal_scraper.requests.get")
@patch("mal_scraper.time.sleep")
def test_search_manga_by_year_missing_data(mock_sleep, mock_get, scraper):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}

    mock_get.return_value = mock_response

    with pytest.raises(KeyError):
        scraper.search_manga_by_year(2020)

# scrape_decade Tests

@patch.object(MALScraper, "search_manga_by_year")
def test_scrape_decade_basic(mock_search, scraper):
    mock_search.return_value = [
        {
            "mal_id": 1,
            "title": "Test Manga",
            "genres": [],
            "themes": [],
            "demographics": [],
            "authors": [],
            "serializations": []
        }
    ]

    result = scraper.scrape_decade(2020, 2020)

    assert isinstance(result, list)
    assert len(result) >= 1
    assert result[0]["title"] == "Test Manga"


@patch.object(MALScraper, "search_manga_by_year")
def test_scrape_decade_no_results(mock_search, scraper):
    mock_search.return_value = []

    result = scraper.scrape_decade(2020, 2020)

    assert result == []


@patch.object(MALScraper, "search_manga_by_year")
def test_scrape_decade_deduplication(mock_search, scraper):
    mock_search.return_value = [
        {"mal_id": 1, "title": "A"},
        {"mal_id": 1, "title": "A"}  # duplicate
    ]

    result = scraper.scrape_decade(2020, 2020)

    ids = [m["manga_id"] for m in result]
    assert len(ids) == len(set(ids))  # no duplicates

# Test save_to_csv works

@patch("mal_scraper.pd.DataFrame.to_csv")
def test_save_to_csv_success(mock_to_csv, scraper):
    data = [{"title": "Berserk"}, {"title": "One Piece"}]

    df = scraper.save_to_csv(data, filename="test.csv")

    assert len(df) == 2
    mock_to_csv.assert_called_once()


def test_save_to_csv_empty(scraper):
    df = scraper.save_to_csv([], filename="empty.csv")

    assert df.empty