import gspread
import pandas as pd


def initialize_google_sheets():
    # Google Sheets API'ye erişim için gerekli izinleri ayarlıyoruz
    client = gspread.service_account(filename="august-oarlock-441317-t9-0a0e6dbc7072.json")

    # Google Sheet'i açıyoruz
    spreadsheet = client.open_by_url(
        'https://docs.google.com/spreadsheets/d/1eBO5jEN6_EMNT0whaCMintfUuvCoOME-WFe6zefjBjU/edit?usp=sharing')

    # İlk sayfayı alıyoruz
    return spreadsheet.get_worksheet(0)


# sheet = initialize_google_sheets()
def sheets_to_df():
    worksheet = initialize_google_sheets()
    # Veriyi çekiyoruz ve pandas DataFrame'e çeviriyoruz
    data = worksheet.get_all_records()  # Bu, veriyi bir liste olarak alır
    df = pd.DataFrame(data)  # Veriyi DataFrame'e çeviriyoruz
    return df


# usage
# write_to_google_sheets(sheet, row_idx, text, references[row_idx - 2], prediction, model_idx, model_name)
def write_to_google_sheets(prompts, chosens, predictions, model_name):
    """
    Google Sheets'e verilen verileri yazar ve modelin sütun indeksini otomatik belirler.

    Args:
        sheet_df (gspread.models.Spreadsheet): Google Sheets dokümanı.
        row_index (int): Yazma işleminin başlayacağı satır numarası.
        prompt (str): Prompt verisi.
        chosen (str): Chosen verisi.
        prediction (str): Prediction (model çıktısı).
        model_name (str): Model adı (ilk satıra yazılır ve sütun indeksini belirler).
    """
    sheet_df = sheets_to_df()
    columns = sheet_df.columns.tolist()
    if "Info" not in columns:
        sheet_df['Info'] = None
    if "Chosen" not in columns:
        sheet_df['Chosen'] = None
    if "Score_1" not in columns:
        sheet_df[f'Score_1'] = None

    for row_index, (prompt, chosen, prediction) in enumerate(zip(prompts, chosens, predictions)):
        if not sheet_df.loc[row_index, "Info"]:
            sheet_df.loc[row_index, "Info"] = prompt
        if not sheet_df.loc[row_index, "Chosen"]:
            sheet_df.loc[row_index, "Chosen"] = chosen
        if not sheet_df.loc[row_index, model_name]:
            sheet_df.loc[row_index, model_name] = prediction
    model_index = sheet_df.columns.get_loc(model_name)
    if f'Score_{model_index}' not in columns:
        sheet_df[f'Score_{model_index}'] = None
    sheet_df = sheet_df.replace([float('nan'), float('inf'), float('-inf')], None)
    headers = sheet_df.columns.tolist()
    updated_data = sheet_df.values.tolist()

    worksheet = initialize_google_sheets()
    # Sheet'in tüm verilerini temizliyoruz
    worksheet.clear()
    worksheet.append_row(headers)
    # Yeni veriyi Sheets'e ekliyoruz
    worksheet.append_rows(updated_data)
    print("Veri başarıyla güncellendi.")