import pytest
import pandas as pd
from unittest.mock import patch

from generate_users import UserInteractionGenerator


@pytest.fixture
def fake_manga():
    return pd.DataFrame({
        "manga_id": [1, 2],
        "title": ["Berserk", "One Piece"],
        "genres": ["Action,Fantasy", "Adventure,Comedy"],
        "score": [9.5, 9.0]
    })


@patch("generate_users.pd.read_csv")
def test_init(mock_read_csv, fake_manga):
    mock_read_csv.return_value = fake_manga

    generator = UserInteractionGenerator()

    assert len(generator.manga_df) == 2
    assert generator.n_users == 5000


@patch("generate_users.pd.read_csv")
def test_generate_user_preferences(mock_read_csv, fake_manga):
    mock_read_csv.return_value = fake_manga

    generator = UserInteractionGenerator()

    generator.n_users = 10

    users = generator.generate_user_preferences()

    assert len(users) == 10
    assert "preferred_genres" in users.columns
    assert "activity_level" in users.columns


@patch("generate_users.pd.read_csv")
def test_generate_interactions(mock_read_csv, fake_manga):
    mock_read_csv.return_value = fake_manga

    generator = UserInteractionGenerator()

    users = pd.DataFrame({
        "user_id": ["user_00001"],
        "preferred_genres": ["Action"],
        "avg_rating": [8.0],
        "activity_level": ["low"]
    })

    interactions = generator.generate_interactions(users)

    assert isinstance(interactions, pd.DataFrame)
    assert len(interactions) > 0

    assert "user_id" in interactions.columns
    assert "manga_id" in interactions.columns
    assert "interaction_type" in interactions.columns
    assert "rating" in interactions.columns


@patch("generate_users.pd.read_csv")
def test_ratings_are_valid(mock_read_csv, fake_manga):
    mock_read_csv.return_value = fake_manga

    generator = UserInteractionGenerator()

    users = pd.DataFrame({
        "user_id": ["user_00001"],
        "preferred_genres": ["Action"],
        "avg_rating": [8.0],
        "activity_level": ["low"]
    })

    interactions = generator.generate_interactions(users)

    assert interactions["rating"].between(1, 10).all()
