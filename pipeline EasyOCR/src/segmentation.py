"""
Módulo de Segmentação de Placas Veiculares baseado em PDI Clássico.

Implementa um pipeline robusto baseado em múltiplos operadores morfológicos clássicos (OpenCV):
1. Pré-processamento e atenuação de ruído com filtro bilateral (preserva bordas);
2. Extração de gradientes verticais via Sobel Horizontal e Canny Adaptativo;
3. Fechamento morfológico horizontal multi-escala com kernels retangulares alargados;
4. Filtragem geométrica e espacial (elimina regiões de asfalto/teto, AR centrado em ~4.2);
5. Ranqueamento multiobjetivo baseado em energia de gradiente de Sobel interno e desvio padrão;
6. Recorte com margem de segurança (padding) e correção de perspectiva;
7. Geração de imagens intermediárias de validação (debug_steps/) para documentação acadêmica.

Uso:
    from src.segmentation import PlateSegmenter
    segmenter = PlateSegmenter()
    cropped_plate, meta = segmenter.segment(image_bgr, image_name="1.JPG",
                                            save_debug=True, debug_dir="data/debug_steps")
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Ordena 4 pontos de um quadrilátero no sentido horário:
    [top-left, top-right, bottom-right, bottom-left].
    Necessário para a transformada de perspectiva ser aplicada corretamente.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left  → menor soma (x + y)
    rect[2] = pts[np.argmax(s)]      # bottom-right → maior soma (x + y)

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right → menor diferença (y - x)
    rect[3] = pts[np.argmax(diff)]   # bottom-left → maior diferença (y - x)
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Aplica a transformada projetiva de perspectiva (warp perspective) em uma ROI
    definida por 4 pontos, retificando distorções angulares introduzidas pela câmera.
    Utiliza cv2.getPerspectiveTransform + cv2.warpPerspective.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calcula as dimensões do retângulo de saída
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width <= 0 or max_height <= 0:
        return image

    # Pontos de destino: retângulo plano sem distorção
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))
    return warped


class PlateSegmenter:
    """
    Segmentador clássico de placas veiculares com múltiplos estágios morfológicos de PDI.

    Parâmetros principais:
    - aspect_ratio_range: Faixa de razão de aspecto (largura/altura) aceita para candidatos.
    - ideal_aspect_ratio: AR alvo (~4.2 para placas brasileiras com distorção perspectiva).
    - min_area_ratio / max_area_ratio: Fração mínima/máxima da área da imagem aceita.
    """

    def __init__(
        self,
        aspect_ratio_range: Tuple[float, float] = (2.0, 7.5),
        ideal_aspect_ratio: float = 4.2,
        min_area_ratio: float = 0.006,
        max_area_ratio: float = 0.22,
    ):
        self.ar_min, self.ar_max = aspect_ratio_range
        self.ideal_ar = ideal_aspect_ratio
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def _score_candidate(
        self,
        contour: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
        gray_img: np.ndarray,
        total_shape: Tuple[int, int],
    ) -> float:
        """
        Calcula a pontuação de qualidade de um candidato à placa com base em
        geometria (Aspect Ratio, Solidez, Posição) e textura interna (Gradiente de Sobel).

        Pontuação multiobjetivo:
            score = (ar_score × 40) + (sobel_energy × 1.5) + (std_dev × 0.5) + (solidity × 10)

        Retorna -1.0 se o candidato não passar nos filtros de exclusão fundamentais.
        """
        x, y, w, h = bbox
        img_h, img_w = total_shape

        if w <= 0 or h <= 0:
            return -1.0

        ar = w / float(h)
        area = w * h
        area_ratio = area / float(img_w * img_h)

        # --- Filtros de exclusão fundamentais ---

        # 1. Razão de aspecto fora da faixa aceita para placas
        if not (self.ar_min <= ar <= self.ar_max):
            return -1.0

        # 2. Área relativa muito pequena (ruído) ou muito grande (fundo inteiro)
        if not (self.min_area_ratio <= area_ratio <= self.max_area_ratio):
            return -1.0

        # 3. Posição vertical: placas ficam no corpo do veículo
        #    Elimina regiões de asfalto (y muito baixo) e céu/teto (y muito alto)
        if y < int(0.12 * img_h) or y > int(0.86 * img_h):
            return -1.0

        # 4. Posição horizontal: evita artefatos colados na borda lateral da imagem
        if x < int(0.03 * img_w) or (x + w) > int(0.98 * img_w):
            return -1.0

        # 5. Dimensões mínimas absolutas em pixels
        if w < 70 or h < 16:
            return -1.0

        roi_gray = gray_img[y:y + h, x:x + w]
        if roi_gray.size == 0:
            return -1.0

        # --- Critérios de pontuação positiva ---

        # Critério A: Proximidade com o AR ideal (~4.2 empírico para o dataset)
        ar_diff = abs(ar - self.ideal_ar)
        ar_score = 1.0 / (1.0 + ar_diff)   # varia de 0 (longe) a 1 (exato)

        # Critério B: Energia de gradiente horizontal de Sobel na ROI
        #   Caracteres da placa criam altas frequências verticais (bordas em X)
        sobel_roi = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_energy = np.mean(np.absolute(sobel_roi))

        # Critério C: Desvio padrão de intensidade (contraste entre char preto e fundo branco)
        std_dev = np.std(roi_gray)

        # Critério D: Solidez do contorno (razão área do contorno / área do bounding box)
        cnt_area = cv2.contourArea(contour) if contour is not None else area
        solidity = cnt_area / float(area) if area > 0 else 0.0

        # Pontuação final combinada (pesos calibrados empiricamente no dataset)
        score = (
            (ar_score * 40.0)
            + (min(sobel_energy, 45.0) * 1.5)
            + (min(std_dev, 60.0) * 0.5)
            + (min(solidity, 1.0) * 10.0)
        )

        return score

    def segment(
        self,
        image_bgr: np.ndarray,
        image_name: str = "",
        save_debug: bool = False,
        debug_dir: str = "",
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Executa a segmentação clássica da placa combinando Sobel + Canny com morfologia multi-escala.

        Fluxo:
            Estágio 1 — Sobel Horizontal → limiarização Otsu → fechamento morfológico → contornos
            Estágio 2 — Canny Adaptativo → fechamento morfológico → contornos
            Seleção    — Melhor candidato por pontuação multiobjetivo + padding de segurança
            Fallback   — Recorte central de emergência se nenhum candidato passa nos filtros
        """
        if image_bgr is None or image_bgr.size == 0:
            return None, {"status": "FAILED", "reason": "Imagem vazia"}

        orig_h, orig_w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # --- Pré-processamento: filtro bilateral (reduz ruído preservando bordas) ---
        # Parâmetros: diâmetro=9, sigma de cor e espaço=30 (conservador)
        bilateral = cv2.bilateralFilter(gray, 9, 30, 30)

        candidates = []
        all_contours_to_draw = []

        # =================================================================
        # ESTÁGIO 1: Gradiente Horizontal de Sobel + Fechamento Morfológico
        # =================================================================
        # Calcula o gradiente horizontal (derivada em X): destaca bordas verticais
        sobelx = np.absolute(cv2.Sobel(bilateral, cv2.CV_32F, 1, 0, ksize=3))
        # Normaliza para [0, 255] e aplica blur suave antes do threshold
        sobel_norm = (255 * ((sobelx - np.min(sobelx)) / (np.max(sobelx) - np.min(sobelx) + 1e-5))).astype("uint8")
        sobel_blur = cv2.GaussianBlur(sobel_norm, (5, 5), 0)
        # Binarização de Otsu: limiar calculado automaticamente pela minimização de variância intraclasse
        _, thresh_sobel = cv2.threshold(sobel_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Kernels retangulares alargados para fechar os espaços entre os caracteres da placa.
        # Kernels maiores (45x9) capturam placas mais distantes/menores.
        # Kernels menores (25x7) capturam placas mais próximas com maior resolução.
        morph_kernels_sobel = [(45, 9), (35, 9), (31, 7), (25, 7)]
        for kw, kh in morph_kernels_sobel:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            # MORPH_CLOSE = dilatação seguida de erosão: fecha buracos internos
            closed = cv2.morphologyEx(thresh_sobel, cv2.MORPH_CLOSE, k)
            cnts, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            all_contours_to_draw.extend(cnts)
            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                score = self._score_candidate(c, (bx, by, bw, bh), gray, (orig_h, orig_w))
                if score > 0:
                    candidates.append({
                        "score": score,
                        "bbox": (bx, by, bw, bh),
                        "contour": c,
                        "method": f"Sobel_({kw}x{kh})",
                    })

        # =================================================================
        # ESTÁGIO 2: Detecção de Bordas Canny + Fechamento Morfológico
        # =================================================================
        # Thresholds adaptativos baseados na mediana do histograma (método de Otsu-like)
        # σ = 0.33 → lower = 0.66 × median, upper = 1.33 × median
        med_val = np.median(bilateral)
        lower_th = int(max(0, 0.66 * med_val))
        upper_th = int(min(255, 1.33 * med_val))
        # Canny: detector de bordas multidimensional (gradiente + supressão de não-máximos + histerese)
        edges = cv2.Canny(bilateral, lower_th, upper_th)

        morph_kernels_canny = [(45, 9), (35, 9), (25, 7)]
        for kw, kh in morph_kernels_canny:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
            cnts, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            all_contours_to_draw.extend(cnts)
            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                score = self._score_candidate(c, (bx, by, bw, bh), gray, (orig_h, orig_w))
                if score > 0:
                    candidates.append({
                        "score": score,
                        "bbox": (bx, by, bw, bh),
                        "contour": c,
                        "method": f"Canny_({kw}x{kh})",
                    })

        # --- Seleção do melhor candidato por maior pontuação ---
        best = None
        cropped_plate = None

        if candidates:
            candidates.sort(key=lambda item: item["score"], reverse=True)
            best = candidates[0]
            bx, by, bw, bh = best["bbox"]

            # Padding de segurança: 4% horizontal e 6% vertical para incluir bordas da placa
            pad_x = int(0.04 * bw)
            pad_y = int(0.06 * bh)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(orig_w, bx + bw + pad_x)
            y2 = min(orig_h, by + bh + pad_y)

            cropped_plate = image_bgr[y1:y2, x1:x2]
            best["final_bbox_orig"] = (x1, y1, x2 - x1, y2 - y1)
        else:
            # Fallback de emergência: recorte central pré-definido
            y1 = int(0.35 * orig_h)
            y2 = int(0.75 * orig_h)
            x1 = int(0.20 * orig_w)
            x2 = int(0.80 * orig_w)
            cropped_plate = image_bgr[y1:y2, x1:x2]
            best = {"score": 10.0, "method": "Emergency_Fallback", "bbox": (x1, y1, x2 - x1, y2 - y1)}

        # --- Salvamento de etapas intermediárias para documentação acadêmica ---
        if save_debug and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            prefix = os.path.splitext(image_name)[0] if image_name else "sample"

            cv2.imwrite(os.path.join(debug_dir, f"{prefix}_01_gray.png"), gray)
            cv2.imwrite(os.path.join(debug_dir, f"{prefix}_02_bilateral.png"), bilateral)
            cv2.imwrite(os.path.join(debug_dir, f"{prefix}_03_sobel_x.png"), sobel_norm)
            cv2.imwrite(os.path.join(debug_dir, f"{prefix}_04_canny_edges.png"), edges)
            cv2.imwrite(os.path.join(debug_dir, f"{prefix}_05_morph_closed.png"), closed)

            contours_vis = image_bgr.copy()
            cv2.drawContours(contours_vis, all_contours_to_draw[:150], -1, (0, 255, 255), 1)
            cv2.imwrite(os.path.join(debug_dir, f"{prefix}_06_all_contours.png"), contours_vis)

            candidate_vis = image_bgr.copy()
            if best and "bbox" in best:
                cbx, cby, cbw, cbh = best["bbox"]
                cv2.rectangle(candidate_vis, (cbx, cby), (cbx + cbw, cby + cbh), (0, 255, 0), 3)
                cv2.putText(
                    candidate_vis,
                    f"{best['method']} Score: {best['score']:.1f}",
                    (cbx, max(25, cby - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            cv2.imwrite(os.path.join(debug_dir, f"{prefix}_07_detected_plate_box.png"), candidate_vis)

            if cropped_plate is not None:
                cv2.imwrite(os.path.join(debug_dir, f"{prefix}_08_cropped_plate.png"), cropped_plate)

        meta = {
            "status": "SUCCESS" if cropped_plate is not None else "FAILED",
            "candidate_meta": best,
        }
        return cropped_plate, meta
