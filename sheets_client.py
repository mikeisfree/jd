import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

def get_sheet_data(spreadsheet_id: str, range_name: str, creds_json_path: str):
    creds = Credentials.from_service_account_file(creds_json_path, scopes=SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(spreadsheet_id)
    worksheet = sheet.worksheet(range_name)
    return worksheet.get_all_records()  # zwraca listę dict
