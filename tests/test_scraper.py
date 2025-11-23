import pytest
from src.scraper import VietlottScraper
import requests
import pandas as pd
import os
from unittest.mock import MagicMock, patch

@pytest.fixture
def scraper():
    return VietlottScraper()

def test_init_defaults(scraper):
    assert scraper.data_file == "data/dataset.csv"
    assert scraper.url == "https://www.vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong"

def test_init_custom():
    scraper = VietlottScraper(data_file="custom.csv", url="http://example.com")
    assert scraper.data_file == "custom.csv"
    assert scraper.url == "http://example.com"

def test_fetch_html_success(scraper, mocker):
    mock_response = mocker.Mock()
    mock_response.text = "<html></html>"
    mock_response.raise_for_status.return_value = None
    mocker.patch("requests.get", return_value=mock_response)

    html = scraper.fetch_html()
    assert html == "<html></html>"
    requests.get.assert_called_once_with(scraper.url, headers=scraper.headers, timeout=10)

def test_fetch_html_failure(scraper, mocker):
    mocker.patch("requests.get", side_effect=requests.RequestException("Error"))
    html = scraper.fetch_html()
    assert html is None

def test_parse_html_valid(scraper):
    html_content = """
    <table class="results-table">
        <tbody>
            <tr>
                <td class="col-date">01/01/2026</td>
                <td class="col-draw-id">00999</td>
                <td class="col-numbers">
                    <span class="num">01</span>
                    <span class="num">02</span>
                    <span class="num">03</span>
                    <span class="num">04</span>
                    <span class="num">05</span>
                </td>
                <td class="col-special"><span class="num">12</span></td>
                <td class="col-jackpot">10.000.000.000</td>
            </tr>
        </tbody>
    </table>
    """
    results = scraper.parse_html(html_content)
    assert results is not None
    assert len(results) == 1
    res = results[0]
    assert res['date'] == "2026-01-01"
    assert res['id'] == "00999"
    assert res['main_1'] == 1
    assert res['main_2'] == 2
    assert res['main_3'] == 3
    assert res['main_4'] == 4
    assert res['main_5'] == 5
    assert res['special'] == 12

def test_parse_html_invalid(scraper):
    html_content = "<html><body></body></html>"
    results = scraper.parse_html(html_content)
    # The code returns None if row not found
    assert results is None

def test_parse_html_none(scraper):
    assert scraper.parse_html(None) is None

def test_update_dataset_create_new(scraper, tmp_path):
    # Use a temporary file path
    test_file = tmp_path / "test_data.csv"
    scraper.data_file = str(test_file)

    new_results = [{
        "date": "2026-01-01",
        "id": "00999",
        "main_1": 1,
        "main_2": 2,
        "main_3": 3,
        "main_4": 4,
        "main_5": 5,
        "special": 12
    }]

    scraper.update_dataset(new_results)

    assert os.path.exists(test_file)
    df = pd.read_csv(test_file)
    assert len(df) == 1
    assert str(df.iloc[0]['id']) == "999" or str(df.iloc[0]['id']) == "00999"

def test_update_dataset_append(scraper, tmp_path):
    test_file = tmp_path / "test_data.csv"
    scraper.data_file = str(test_file)

    # Create existing file
    initial_data = pd.DataFrame([{
        'date': '2025-12-31', 'id': '00998',
        'result/0': 1, 'result/1': 2, 'result/2': 3, 'result/3': 4, 'result/4': 5, 'db': 10
    }])
    initial_data.to_csv(test_file, index=False)

    new_results = [{
        "date": "2026-01-01",
        "id": "00999",
        "main_1": 6,
        "main_2": 7,
        "main_3": 8,
        "main_4": 9,
        "main_5": 10,
        "special": 11
    }]

    scraper.update_dataset(new_results)

    df = pd.read_csv(test_file)
    assert len(df) == 2
    assert str(df.iloc[1]['id']) == "999" or str(df.iloc[1]['id']) == "00999"

def test_update_dataset_skip_existing(scraper, tmp_path):
    test_file = tmp_path / "test_data.csv"
    scraper.data_file = str(test_file)

    # Create existing file
    initial_data = pd.DataFrame([{
        'date': '2026-01-01', 'id': '00999',
        'result/0': 1, 'result/1': 2, 'result/2': 3, 'result/3': 4, 'result/4': 5, 'db': 12
    }])
    initial_data.to_csv(test_file, index=False)

    new_results = [{
        "date": "2026-01-01",
        "id": "00999",
        "main_1": 1,
        "main_2": 2,
        "main_3": 3,
        "main_4": 4,
        "main_5": 5,
        "special": 12
    }]

    scraper.update_dataset(new_results)

    df = pd.read_csv(test_file)
    assert len(df) == 1 # No duplicates added
