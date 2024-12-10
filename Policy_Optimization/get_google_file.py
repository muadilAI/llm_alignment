import gspread
# from oauth2client.service_account import ServiceAccountCredentials

# usage


# sheet = initialize_google_sheets()
def initialize_google_sheets():
    # Google Sheets bağlantısını başlat
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    # credentials = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile_name, scope)
    # print(credentials)
    # client = gspread.authorize(credentials)
    gc = gspread.service_account(filename ="tests/august-oarlock-441317-t9-0a0e6dbc7072.json")
    # Google Sheets dokümanını aç
    spreadsheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1in0Xvs-rFgF_ORVnfbKhu_LElumt1dKPeZ01kLtF4XY/edit?usp=sharing")
    # Çalışma sayfasını seç (örneğin ilk sayfa)
    sheet = spreadsheet.get_worksheet(0)  # İlk çalışma sayfası
    return sheet

# usage
# write_to_google_sheets(sheet, row_idx, text, references[row_idx - 2], prediction, model_idx, model_name)
def write_to_google_sheets(sheet, row_index, prompt, chosen, prediction, model_name):
    """
    Google Sheets'e verilen verileri yazar ve modelin sütun indeksini otomatik belirler.

    Args:
        sheet (gspread.models.Spreadsheet): Google Sheets dokümanı.
        row_index (int): Yazma işleminin başlayacağı satır numarası.
        prompt (str): Prompt verisi.
        chosen (str): Chosen verisi.
        prediction (str): Prediction (model çıktısı).
        model_name (str): Model adı (ilk satıra yazılır ve sütun indeksini belirler).
    """
    # İlk satırdaki sütunları kontrol et
    first_row = sheet.row_values(1)
    model_index = None

    # Model adını sütunda bul
    if model_name in first_row:
        model_index = first_row.index(model_name) + 1  # Sütun indeksi (1 bazlı)
    else:
        # Son dolu sütunun 2 sütun sonrasını yeni model sütunu olarak ayarla
        model_index = len(first_row) + 2
        # Yeni model adını ilk satıra yaz
        sheet.update_cell(1, model_index, model_name)

    # Prompt, chosen ve prediction verilerini yaz
    if not sheet.cell(row_index, 1).value:  # Prompt hücresi boşsa yaz
        sheet.update_cell(row_index, 1, prompt)
    if not sheet.cell(row_index, 2).value:  # Chosen hücresi boşsa yaz
        sheet.update_cell(row_index, 2, chosen)
    if not sheet.cell(row_index, model_index).value:  # Prediction hücresi boşsa yaz
        sheet.update_cell(row_index, model_index, prediction)


