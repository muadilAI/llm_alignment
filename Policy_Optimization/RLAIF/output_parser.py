
from typing import Optional

class OutputParser:
    def __init__(self, model, sheet):
        """
        OutputParser sınıfı.

        Args:
            model: Llama model nesnesi.
            sheet (Worksheet): Çalışma sayfası nesnesi (gspread Worksheet).
        """
        self.model = model
        self.sheet = sheet

    def compare_summaries(self, prompt: str, chosen: str, predicted: str) -> int:
        """
        Modelden kıyaslama sonucu alır ve skoru döndürür.

        Args:
            prompt (str): Orijinal prompt.
            chosen (str): Kullanıcının seçtiği summary.
            predicted (str): Modelin tahmin ettiği summary.

        Returns:
            int: 1 (chosen daha iyi) veya 2 (predicted daha iyi).
        """
        # Llama modeline input hazırlığı
        input_text = (
            f"Given the following prompt:\n{prompt}\n\n"
            f"Compare the following summaries:\n"
            f"1. Chosen summary: {chosen}\n"
            f"2. Predicted summary: {predicted}\n\n"
            "Which summary is better? Respond with '1' for Chosen or '2' for Predicted."
        )

        # Modeli çağır ve cevabı al
        response = self.model.generate(input_text)
        output = response.strip()

        # Çıktıyı parse et
        if output == "1":
            return 1  # Chosen daha iyi
        elif output == "2":
            return 2  # Predicted daha iyi
        else:
            raise ValueError(f"Unexpected model output: {output}")

    def update_sheet_with_score(self, row_index: int, score: int, score_column_index: int):
        """
        Hesaplanan skoru çalışma sayfasına yazar.

        Args:
            row_index (int): Skorun yazılacağı satır indeksi.
            score (int): Hesaplanan skor.
            score_column_index (int): Skorun yazılacağı sütun indeksi.
        """
        self.sheet.update_cell(row_index, score_column_index, score)

    def process_row(self, row_index: int, prompt_column: int, chosen_column: int, predicted_column: int, score_column: int):
        """
        Belirli bir satır için işlemleri gerçekleştirir.

        Args:
            row_index (int): İşlenecek satır indeksi.
            prompt_column (int): Prompt'un bulunduğu sütun.
            chosen_column (int): Chosen summary'nin bulunduğu sütun.
            predicted_column (int): Predicted summary'nin bulunduğu sütun.
            score_column (int): Skorun yazılacağı sütun.
        """
        # Hücrelerden verileri al
        prompt = self.sheet.cell(row_index, prompt_column).value
        chosen = self.sheet.cell(row_index, chosen_column).value
        predicted = self.sheet.cell(row_index, predicted_column).value

        # Boş değer kontrolü
        if not prompt or not chosen or not predicted:
            print(f"Row {row_index}: Missing data, skipping.")
            return

        # Kıyaslama yap ve skoru al
        try:
            score = self.compare_summaries(prompt, chosen, predicted)
            print(f"Row {row_index}: Score computed as {score}.")
        except ValueError as e:
            print(f"Row {row_index}: Error during comparison - {e}")
            return

        # Skoru çalışma sayfasına yaz
        self.update_sheet_with_score(row_index, score, score_column)
