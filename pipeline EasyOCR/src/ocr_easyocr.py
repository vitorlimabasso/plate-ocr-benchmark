"""
Módulo de Reconhecimento Óptico de Caracteres usando EasyOCR.

Implementa o pipeline de OCR baseado em Deep Learning:
- Detector de texto: CRAFT (Character Region Awareness For Text Detection)
  → Rede CNN que prediz mapas de calor de regiões e afinidade de caracteres.
- Reconhecedor de sequência: CRNN com backbone ResNet
  → Converte features visuais em sequências de caracteres via CTC decoding.

Diferenças em relação ao Tesseract:
- Não exige binarização: opera em imagens coloridas (BGR) com contraste CLAHE.
- Detecta texto em múltiplos blocos e ordena as detecções da esquerda para a direita.
- Mais robusto a fontes estilizadas (FE-Schrift das placas alemãs/Mercosul) e
  a variações de perspectiva não totalmente corrigidas.

Dependências:
    pip install easyocr torch torchvision
    # Os modelos são baixados automaticamente em ~/.EasyOCR/model/ na primeira execução.
"""

import os
import time
import numpy as np
from typing import Dict, Any, List, Optional

try:
    import easyocr
    import torch
except ImportError:
    easyocr = None
    torch = None

from src.utils import parse_and_validate_plate, clean_ocr_text


class EasyOCRPipeline:
    """
    Pipeline de OCR com EasyOCR para reconhecimento de placas veiculares.

    O Reader é instanciado apenas uma vez e mantido em memória para evitar
    overhead de carregamento dos modelos CRAFT + CRNN a cada imagem.
    """

    def __init__(self, languages: Optional[List[str]] = None, use_gpu: Optional[bool] = None):
        """
        Inicializa o EasyOCR Reader.

        Args:
            languages: Lista de códigos de idiomas ISO 639-1 (padrão: ['en']).
            use_gpu:   True = força CUDA; False = CPU; None = detecta automaticamente.
        """
        if easyocr is None:
            raise ImportError("O pacote easyocr ou torch não está instalado.")

        if languages is None:
            languages = ["en"]

        # Detecta disponibilidade de GPU CUDA automaticamente
        if use_gpu is None:
            use_gpu = torch.cuda.is_available() if torch else False

        self.use_gpu = use_gpu
        self.languages = languages
        # Whitelist: apenas maiúsculas e dígitos — reduz falsos positivos
        self.whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        # Carrega os modelos CRAFT e CRNN uma única vez
        # model_storage_directory: pasta local para cache dos pesos (.pth)
        self.reader = easyocr.Reader(
            self.languages,
            gpu=self.use_gpu,
            verbose=False,
            model_storage_directory=os.path.expanduser("~/.EasyOCR/model"),
        )

    def recognize(self, image: np.ndarray, image_name: str = "") -> Dict[str, Any]:
        """
        Executa a detecção e o reconhecimento de texto na ROI da placa.

        Recebe: imagem BGR com contraste reforçado (CLAHE no canal L do espaço LAB).
        Retorna: dicionário com texto bruto, placa extraída, padrão e métricas.

        O CRAFT detecta caixas de texto em múltiplos blocos; esta função os
        ordena pela coordenada X do canto superior esquerdo (esquerda → direita)
        antes de concatená-los em um único token de placa.
        """
        t_start = time.perf_counter()
        raw_text = ""
        confidences = []

        try:
            # readtext retorna: [ (bbox_4pts, text_string, prob_float), ... ]
            # detail=1 → inclui coordenadas e probabilidade de cada detecção
            # paragraph=False → não agrupa linhas em parágrafos
            results = self.reader.readtext(
                image,
                allowlist=self.whitelist,
                detail=1,
                paragraph=False,
            )

            if results:
                # Ordena os blocos detectados pela coordenada X inicial do bbox
                # (canto superior esquerdo do quadrilátero retornado pelo CRAFT)
                sorted_results = sorted(results, key=lambda item: item[0][0][0])

                tokens = []
                for _, text, prob in sorted_results:
                    tokens.append(text)
                    confidences.append(prob)

                # Concatena todos os tokens sem separador (placa = string contínua)
                raw_text = "".join(tokens)
            else:
                raw_text = ""

        except Exception as e:
            raw_text = f"ERRO_OCR: {str(e)}"

        t_elapsed = time.perf_counter() - t_start

        # Validação estrutural do padrão da placa
        parsed = parse_and_validate_plate(raw_text)

        mean_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "image_name":    image_name,
            "engine":        "EasyOCR",
            "raw_text":      raw_text.strip(),
            "cleaned_text":  parsed["cleaned_text"],
            "plate":         parsed["plate"],
            "pattern":       parsed["pattern"],
            "is_valid":      parsed["is_valid"],
            "confidence":    round(mean_confidence, 2),
            "execution_time": t_elapsed,
            "gpu_used":      self.use_gpu,
        }
