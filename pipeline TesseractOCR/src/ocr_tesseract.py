"""
Módulo de Reconhecimento Óptico de Caracteres usando Pytesseract (Tesseract OCR).

Implementa o pipeline de OCR com Tesseract adaptado para placas veiculares brasileiras:
- Localização automática do executável tesseract.exe no Windows (PATH + caminhos padrão);
- Parametrização PSM (Page Segmentation Mode) com fallback adaptativo multi-estágio;
- Whitelist restrita de caracteres alfanuméricos maiúsculos (A-Z, 0-9);
- Estratégia de sub-região: recorte dos 78% inferiores da placa para ignorar
  o banner azul "BRASIL" das placas Mercosul, que confunde o PSM 7.

Dependências:
    pip install pytesseract
    # Tesseract deve estar instalado separadamente:
    # Windows: https://github.com/UB-Mannheim/tesseract/wiki
    # Linux:   sudo apt install tesseract-ocr
"""

import os
import time
import shutil
import cv2
import numpy as np
from typing import Dict, Any, Optional

try:
    import pytesseract
except ImportError:
    pytesseract = None

from src.utils import parse_and_validate_plate, clean_ocr_text


def configure_tesseract_cmd() -> str:
    """
    Localiza e configura o caminho do executável tesseract.exe.

    Estratégia de busca (ordem de prioridade):
    1. shutil.which('tesseract') → usa o PATH do sistema (Linux/Mac/Windows com PATH correto)
    2. Caminhos padrão de instalação no Windows (Program Files, LocalAppData)
    3. Mantém o valor atual de pytesseract.tesseract_cmd como último recurso

    Retorna o caminho configurado ou o valor padrão do pytesseract.
    """
    if pytesseract is None:
        raise ImportError("O pacote pytesseract não está instalado.")

    # Tentativa 1: usar PATH do sistema operacional
    which_tess = shutil.which("tesseract")
    if which_tess:
        pytesseract.pytesseract.tesseract_cmd = which_tess
        return which_tess

    # Tentativa 2: caminhos padrão de instalação no Windows
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            pytesseract.pytesseract.tesseract_cmd = cand
            return cand

    # Mantém o valor padrão do pytesseract
    return pytesseract.pytesseract.tesseract_cmd


class TesseractOCRPipeline:
    """
    Pipeline de OCR com Tesseract para reconhecimento de placas veiculares brasileiras.

    Configuração do motor:
    - OEM 3: LSTM + legado combinados (melhor acurácia geral)
    - PSM 7: Trata a imagem como uma única linha de texto (padrão para placas)
    - Whitelist: apenas letras maiúsculas A-Z e dígitos 0-9

    Fallback adaptativo (em cascata):
        PSM 7 → PSM 6 → recorte inferior 78% (ignora banner BRASIL) → PSM 8
    """

    def __init__(self, lang: str = "eng", psm: int = 7):
        self.cmd_path = configure_tesseract_cmd()
        self.lang = lang
        self.psm = psm
        self.whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.base_config = f"--oem 3 --psm {self.psm} -c tessedit_char_whitelist={self.whitelist}"

    def recognize(self, image: np.ndarray, image_name: str = "") -> Dict[str, Any]:
        """
        Executa o reconhecimento de texto na imagem binarizada da placa.

        Recebe: imagem em escala de cinza binarizada (Otsu) com fundo branco e texto preto.
        Retorna: dicionário com texto bruto, texto limpo, placa extraída, padrão e métricas.
        """
        if pytesseract is None:
            raise RuntimeError("pytesseract não está disponível.")

        t_start = time.perf_counter()
        raw_text = ""
        used_psm = self.psm

        try:
            # Tentativa 1 — PSM 7: imagem = linha única de texto
            raw_text = pytesseract.image_to_string(image, lang=self.lang, config=self.base_config)
            cleaned = clean_ocr_text(raw_text)

            # Tentativa 2 — PSM 6: imagem = bloco uniforme de texto (captura mais caracteres)
            if len(cleaned) < 5:
                cfg_psm6 = f"--oem 3 --psm 6 -c tessedit_char_whitelist={self.whitelist}"
                raw_6 = pytesseract.image_to_string(image, lang=self.lang, config=cfg_psm6)
                clean_6 = clean_ocr_text(raw_6)
                if len(clean_6) > len(cleaned):
                    raw_text, cleaned, used_psm = raw_6, clean_6, 6

            # Tentativa 3 — Recorte inferior (78% da altura): descarta banner azul "BRASIL"
            # Placas Mercosul têm o texto de identificação do país na faixa superior (~22% da altura)
            if len(cleaned) < 5:
                h, w = image.shape[:2]
                if h >= 30:
                    char_roi = image[int(0.22 * h):, :]
                    cfg_char = f"--oem 3 --psm 7 -c tessedit_char_whitelist={self.whitelist}"
                    raw_char = pytesseract.image_to_string(char_roi, lang=self.lang, config=cfg_char)
                    clean_char = clean_ocr_text(raw_char)
                    if len(clean_char) > len(cleaned):
                        raw_text, cleaned, used_psm = raw_char, clean_char, 7

            # Tentativa 4 — PSM 8: imagem = palavra única
            if len(cleaned) < 5:
                cfg_psm8 = f"--oem 3 --psm 8 -c tessedit_char_whitelist={self.whitelist}"
                raw_8 = pytesseract.image_to_string(image, lang=self.lang, config=cfg_psm8)
                clean_8 = clean_ocr_text(raw_8)
                if len(clean_8) > len(cleaned):
                    raw_text, cleaned, used_psm = raw_8, clean_8, 8

        except Exception as e:
            raw_text = f"ERRO_OCR: {str(e)}"

        t_elapsed = time.perf_counter() - t_start

        # Validação estrutural do padrão da placa (Mercosul / Antiga)
        parsed = parse_and_validate_plate(raw_text)

        # Extrai confiança média reportada pelo Tesseract (via image_to_data)
        confidence = 0.0
        try:
            data = pytesseract.image_to_data(
                image, lang=self.lang, config=self.base_config, output_type=pytesseract.Output.DICT
            )
            confs = [float(c) for c in data.get("conf", []) if float(c) > 0]
            if confs:
                confidence = sum(confs) / len(confs)
        except Exception:
            confidence = 0.0

        return {
            "image_name":    image_name,
            "engine":        "Tesseract",
            "raw_text":      raw_text.strip(),
            "cleaned_text":  parsed["cleaned_text"],
            "plate":         parsed["plate"],
            "pattern":       parsed["pattern"],
            "is_valid":      parsed["is_valid"],
            "confidence":    round(confidence, 2),
            "execution_time": t_elapsed,
            "psm_used":      used_psm,
        }
