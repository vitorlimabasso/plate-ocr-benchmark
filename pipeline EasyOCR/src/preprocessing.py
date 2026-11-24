"""
Módulo de Pré-processamento de Placas Veiculares para Otimização de OCR.

Aplica uma cadeia de técnicas clássicas de PDI na ROI recortada da placa.
Cada técnica foi testada e calibrada para maximizar a legibilidade dos caracteres:

Cadeia de processamento:
    1. Redimensionamento bicúbico → altura padrão de 120 px (preserva proporção)
    2. Conversão para escala de cinza
    3. Filtro bilateral → suavização de ruído preservando bordas dos caracteres
    4. CLAHE (Contrast Limited Adaptive Histogram Equalization) → melhora contraste local
    5. Binarização de Otsu → separação binária texto/fundo com limiar ótimo automático
    6. Correção de polaridade → garante fundo branco e texto preto (exigência do Tesseract)
    7. Versão LAB-CLAHE colorida → imagem BGR melhorada para o EasyOCR

Uso:
    from src.preprocessing import preprocess_plate
    preprocessed = preprocess_plate(plate_bgr, target_height=120)
    # Acesse: preprocessed["binary"], preprocessed["enhanced_bgr"], etc.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict


def resize_to_standard_height(image: np.ndarray, target_height: int = 120) -> np.ndarray:
    """
    Redimensiona a placa mantendo a proporção de aspecto (aspect ratio).

    Técnica: Interpolação bicúbica (cv2.INTER_CUBIC).
    Por que bicúbica? Usa um polinômio de grau 3 para suavizar transições de intensidade,
    preservando os traços finos dos caracteres sem os artefatos de pixelização da interpolação
    bilinear e sem o excessivo borramento da interpolação de Lanczos.
    Altura-alvo de 120 px empírica: resolve caracteres legíveis com largura ~504-600 px.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image

    scale = target_height / float(h)
    new_width = int(w * scale)
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
    return resized


def apply_clahe(gray_image: np.ndarray, clip_limit: float = 2.5, tile_grid: Tuple[int, int] = (4, 4)) -> np.ndarray:
    """
    Aplica CLAHE — Equalização de Histograma Adaptativa Limitada por Contraste.

    Diferença em relação à equalização global:
    - A equalização global aplica um único mapeamento de histograma à imagem inteira,
      amplificando ruído em regiões já uniformes.
    - O CLAHE divide a imagem em tiles (blocos) de tamanho tile_grid e equaliza
      cada tile individualmente, depois interpola as bordas (interpolação bilinear).
    - O parâmetro clip_limit limita o ganho máximo em cada bin, evitando
      superamplificação de ruído (artefatos de Gibbs).

    Resultado: ilumina regiões em sombra e suaviza reflexos sem perder textura dos chars.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray_image)


def correct_polarity(binary_image: np.ndarray) -> np.ndarray:
    """
    Garante que o texto esteja escuro (0) sobre fundo claro (255) — padrão do Tesseract.

    Estratégia: Mede a intensidade média nos 5 pixels de borda de cada lado da imagem.
    Hipótese: o fundo da placa domina as bordas. Se a borda média < 127, a imagem está
    invertida (texto branco em fundo preto) e deve ser negada com cv2.bitwise_not.

    Testadas e descartadas:
    - Contagem de pixels blancos vs. pretos: sensível à densidade de caracteres.
    - Análise do modo do histograma: instável em imagens ruidosas.
    """
    h, w = binary_image.shape[:2]
    border_pixels = np.concatenate([
        binary_image[0:5, :].flatten(),
        binary_image[h - 5:h, :].flatten(),
        binary_image[:, 0:5].flatten(),
        binary_image[:, w - 5:w].flatten(),
    ])
    mean_val = np.mean(border_pixels)
    # Borda escura (< 127) → fundo preto → inverte para fundo branco
    if mean_val < 127:
        return cv2.bitwise_not(binary_image)
    return binary_image


def preprocess_plate(
    plate_bgr: np.ndarray,
    target_height: int = 120,
    save_debug: bool = False,
    debug_prefix: str = "",
    debug_dir: str = "",
) -> Dict[str, np.ndarray]:
    """
    Pipeline completo de pré-processamento de PDI da placa recortada.

    Retorna um dicionário com todas as versões intermediárias da imagem:
    - 'original'     : Placa original recortada (sem processamento)
    - 'resized'      : Placa redimensionada (interpolação bicúbica)
    - 'gray'         : Escala de cinza (cv2.COLOR_BGR2GRAY)
    - 'denoised'     : Suavização bilateral (atenua ruído preservando bordas)
    - 'clahe'        : Equalização adaptativa de contraste (CLAHE)
    - 'binary'       : Binarização de Otsu + correção de polaridade (ideal para Tesseract)
    - 'enhanced_bgr' : BGR com contraste CLAHE no canal L (espaço LAB, ideal para EasyOCR)

    Nota: Tesseract recebe 'binary' (1 canal, binarizado).
          EasyOCR recebe 'enhanced_bgr' (3 canais, colorido e contrastado).
    """
    if plate_bgr is None or plate_bgr.size == 0:
        raise ValueError("Imagem da placa inválida ou vazia.")

    # Passo 1 — Redimensionamento bicúbico para altura padrão
    resized = resize_to_standard_height(plate_bgr, target_height=target_height)

    # Passo 2 — Conversão para tons de cinza (canal único)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Passo 3 — Filtro bilateral para redução de ruído de textura
    # d=7: diâmetro da vizinhança; sigmaColor=50: tolerância de cor; sigmaSpace=50: tolerância espacial
    # Valor de sigma conservador (50) para não borrar bordas dos caracteres
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Passo 4 — CLAHE: equalização adaptativa para compensar variações de iluminação e reflexos
    # clip_limit=2.5 e tile_grid=(4,4): balanço entre contraste e ruído testado no dataset
    clahe_img = apply_clahe(denoised, clip_limit=2.5, tile_grid=(4, 4))

    # Passo 5 — Binarização pelo Método de Otsu
    # Otsu encontra o limiar t* que minimiza a variância intraclasse dos dois grupos (fundo/texto)
    # THRESH_BINARY + THRESH_OTSU: thresh=0 é ignorado; o valor ótimo é calculado automaticamente
    # Testado e descartado: threshold adaptativo (adaptive Gaussian) — mais sensível a reflexos locais
    _, binary = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Passo 6 — Correção de polaridade (texto preto / fundo branco exigido pelo Tesseract)
    binary_corrected = correct_polarity(binary)

    # Passo 7 — Versão colorida melhorada para EasyOCR (opera com redes neurais em 3 canais)
    # Converte para LAB, aplica CLAHE apenas no canal L (luminosidade) e reconverte para BGR
    # Por que LAB? Desacopla luminosidade de crominância: CLAHE só afeta brilho, não cor
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = apply_clahe(l, clip_limit=2.0, tile_grid=(4, 4))
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Salvamento opcional das etapas intermediárias para documentação
    if save_debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_prep_01_resized.png"),       resized)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_prep_02_gray.png"),          gray)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_prep_03_denoised.png"),      denoised)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_prep_04_clahe.png"),         clahe_img)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_prep_05_binary.png"),        binary_corrected)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_prep_06_enhanced_bgr.png"), enhanced_bgr)

    return {
        "original":     plate_bgr,
        "resized":      resized,
        "gray":         gray,
        "denoised":     denoised,
        "clahe":        clahe_img,
        "binary":       binary_corrected,
        "enhanced_bgr": enhanced_bgr,
    }
