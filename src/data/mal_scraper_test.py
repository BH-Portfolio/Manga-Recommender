import pytest
from unittest import patch, MagicMock
import mal_scraper

@pytest.fixture
def scraper():
    return MALScraper()
