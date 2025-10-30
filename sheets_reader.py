# sheets_reader.py
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def read_sheet(sheet_id, range_name="Sheet1"):
    # Requires service account JSON credentials in env var GOOGLE_SA_JSON or file
    creds_json = os.environ.get("GOOGLE_SA_JSON_PATH")
    if not creds_json:
        raise RuntimeError("Set GOOGLE_SA_JSON_PATH to path of service account json")
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_json, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).sheet1
    rows = sheet.get_all_records()
    return rows
