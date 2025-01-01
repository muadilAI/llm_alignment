import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# Google Sheets API'ye erişim için gerekli izinleri ayarlıyoruz
client = gspread.service_account(filename="august-oarlock-441317-t9-0a0e6dbc7072.json")

# Google Sheet'i açıyoruz
spreadsheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1eBO5jEN6_EMNT0whaCMintfUuvCoOME-WFe6zefjBjU/edit?usp=sharing')

# İlk sayfayı alıyoruz
worksheet = spreadsheet.get_worksheet(0)

# Veriyi çekiyoruz ve pandas DataFrame'e çeviriyoruz
data = worksheet.get_all_records()  # Bu, veriyi bir liste olarak alır
df = pd.DataFrame(data)  # Veriyi DataFrame'e çeviriyoruz

# DataFrame üzerinde değişiklikler yapıyoruz
# Örnek: Yeni bir sütun eklemek
# df['1'] = 'Yeni Veri'

# Örnek: Belirli bir satıra veri eklemek
df["Model 1"] = None
if df.loc[0, "Model 1"]:
    df.loc[0, "Model 1"] = 'sveri'


# NaN veya inf değerlerini temizleme
df = df.replace([float('nan'), float('inf'), float('-inf')], None)
print(df)
# Veriyi Google Sheets formatına uygun hale getirmek için listeye çeviriyoruz
updated_data = df.values.tolist()
headers = df.columns.tolist()
# Sheet'in tüm verilerini temizliyoruz
worksheet.clear()
worksheet.append_row(headers)
# Yeni veriyi Sheets'e ekliyoruz
worksheet.append_rows(updated_data)

print("Veri başarıyla güncellendi.")


