import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VietlottScraper:
    def __init__(self, data_file="data/dataset.csv", url=None):
        self.data_file = data_file
        # Default to a placeholder URL if not provided.
        # In a real scenario, this would be the actual Vietlott results page.
        self.url = url or "https://www.vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; LottoAI/1.0)"
        }

    def fetch_html(self):
        """Fetches the HTML content from the target URL."""
        try:
            logger.info(f"Fetching data from {self.url}...")
            response = requests.get(self.url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch data: {e}")
            return None

    def parse_html(self, html_content):
        """
        Parses the HTML to extract lottery results.

        Note: The selectors used here are based on the provided design pattern.
        Since 'Lotto 5/35' is a future/hypothetical game, these selectors
        are illustrative and would need to be adjusted for the actual production site.
        """
        if not html_content:
            return None

        soup = BeautifulSoup(html_content, 'html.parser')
        results = []

        # Example: Find the results table (Adjust selector for real site)
        # Using the logic from the user's "porth": table.results-table tbody tr:first-child

        # Hypothetical parsing logic:
        # We iterate over rows to find new results not in our DB
        # For MVP, we just try to parse the 'latest' row

        try:
            # Placeholder selectors - assume a standard table structure
            row = soup.select_one("table.results-table tbody tr:first-child")

            if not row:
                logger.warning("No results table found in HTML. Selector might be outdated.")
                return None

            # Extract fields
            date_str = row.select_one("td.col-date").get_text(strip=True)
            draw_id = row.select_one("td.col-draw-id").get_text(strip=True)

            # Main numbers (01-35)
            # Assuming they are in spans inside a specific cell
            main_nums_els = row.select("td.col-numbers span.num")
            main_numbers = [int(el.get_text(strip=True)) for el in main_nums_els[:5]]

            # Special number (01-12)
            special_num_el = row.select_one("td.col-special span.num")
            special_number = int(special_num_el.get_text(strip=True)) if special_num_el else 0

            # Format date (Assume DD/MM/YYYY)
            # In dataset.csv, format is YYYY-MM-DD
            dt_obj = datetime.strptime(date_str, "%d/%m/%Y")
            formatted_date = dt_obj.strftime("%Y-%m-%d")

            result = {
                "date": formatted_date,
                "id": draw_id,
                "main_1": main_numbers[0],
                "main_2": main_numbers[1],
                "main_3": main_numbers[2],
                "main_4": main_numbers[3],
                "main_5": main_numbers[4],
                "special": special_number
            }
            results.append(result)
            logger.info(f"Parsed result: {result}")
            return results

        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return None

    def update_dataset(self, new_results):
        """Appends new results to the CSV file if they don't exist."""
        if not new_results:
            return

        if os.path.exists(self.data_file):
            df = pd.read_csv(self.data_file)
        else:
            # Create new if doesn't exist
            df = pd.DataFrame(columns=['date', 'id', 'result/0', 'result/1', 'result/2', 'result/3', 'result/4', 'db'])

        # Rename columns to match internal logic if needed, or map back to CSV schema
        # CSV Schema: date,id,result/0,result/1,result/2,result/3,result/4,db

        updates_count = 0
        for res in new_results:
            # Check if ID exists
            # Try to normalize to int for comparison to handle "00999" vs 999 mismatch
            try:
                # Check against integer values
                existing_ids = df['id'].fillna(-1).astype(int).values
                if int(res['id']) in existing_ids:
                    logger.info(f"Draw {res['id']} already exists. Skipping.")
                    continue
            except (ValueError, TypeError):
                # Fallback to exact string comparison if non-numeric or None
                if str(res['id']) in df['id'].astype(str).values:
                    logger.info(f"Draw {res['id']} already exists. Skipping.")
                    continue

            # Add new row
            new_row = {
                'date': res['date'],
                'id': res['id'],
                'result/0': res['main_1'],
                'result/1': res['main_2'],
                'result/2': res['main_3'],
                'result/3': res['main_4'],
                'result/4': res['main_5'],
                'db': res['special']
            }
            # Append using loc or concat
            # Using list of dicts to concat is efficient
            new_row_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_row_df], ignore_index=True)
            updates_count += 1

        if updates_count > 0:
            df.to_csv(self.data_file, index=False)
            logger.info(f"Successfully added {updates_count} new records to {self.data_file}")
        else:
            logger.info("No new unique records found.")

    def run(self):
        """Main execution method."""
        html = self.fetch_html()
        if html:
            data = self.parse_html(html)
            self.update_dataset(data)

if __name__ == "__main__":
    # Test run
    scraper = VietlottScraper()
    # Mocking HTML for testing purposes since the real site doesn't have 5/35 yet
    mock_html = """
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
    print("Testing parser with mock HTML...")
    parsed = scraper.parse_html(mock_html)
    print("Parsed Data:", parsed)
    # scraper.update_dataset(parsed) # Uncomment to actually write to CSV
