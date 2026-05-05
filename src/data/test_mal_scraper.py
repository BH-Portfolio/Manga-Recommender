import pytest
from unittest.mock import patch, MagicMock
from mal_scraper import MALScraper

@pytest.fixture
def scraper():
    return MALScraper()

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
