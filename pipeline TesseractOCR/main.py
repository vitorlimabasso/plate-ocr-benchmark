"""
Pipeline de ALPR com Tesseract OCR.

Orquestra de ponta a ponta:
  1. Leitura das imagens em data/raw/
  2. Segmentação clássica da placa por PDI (Sobel + Canny + Morfologia)
  3. Pré-processamento da ROI (bilateral + CLAHE + Otsu + polaridade)
  4. Reconhecimento com Tesseract OCR (imagem binarizada)
  5. Salvamento do resultado em results/resultados_tesseract.txt

Uso:
    python main.py
    python main.py --raw-dir data/raw --no-save-debug
"""

import os
import sys
import glob
import argparse
import cv2

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.segmentation import PlateSegmenter
from src.preprocessing import preprocess_plate
from src.ocr_tesseract import TesseractOCRPipeline
from src.utils import save_ocr_results


def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline Tesseract OCR — Segmentação de Placas")
    parser.add_argument("--raw-dir",       default=os.path.join("data", "raw"))
    parser.add_argument("--segmented-dir", default=os.path.join("data", "segmented"))
    parser.add_argument("--debug-dir",     default=os.path.join("data", "debug_steps"))
    parser.add_argument("--results-dir",   default="results")
    parser.add_argument("--save-debug",    action="store_true", default=True)
    parser.add_argument("--no-save-debug", dest="save_debug", action="store_false")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 70)
    print("PIPELINE TESSERACT OCR — Segmentação e Reconhecimento de Placas")
    print("=" * 70)

    os.makedirs(args.segmented_dir, exist_ok=True)
    os.makedirs(args.debug_dir,     exist_ok=True)
    os.makedirs(args.results_dir,   exist_ok=True)

    # Localiza imagens em data/raw/
    extensions = ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.JPEG","*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.raw_dir, ext)))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"[ERRO] Nenhuma imagem encontrada em '{args.raw_dir}'.")
        print("Coloque as imagens do dataset em data/raw/ e execute novamente.")
        sys.exit(1)

    print(f"\n[INFO] {len(image_paths)} imagens encontradas em '{args.raw_dir}'.\n")

    # Inicialização dos módulos
    segmenter  = PlateSegmenter()
    ocr_engine = TesseractOCRPipeline(lang="eng", psm=7)

    results = []
    total   = len(image_paths)

    for idx, img_path in enumerate(image_paths, 1):
        filename  = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        print(f"[{idx:02d}/{total:02d}] {filename}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("  [AVISO] Falha ao carregar. Pulando.")
            continue

        # Etapa 1 — Segmentação da placa por PDI clássico
        cropped, _ = segmenter.segment(
            image_bgr=img_bgr,
            image_name=filename,
            save_debug=args.save_debug,
            debug_dir=args.debug_dir,
        )

        if cropped is None or cropped.size == 0:
            print("  [FALHA] Segmentação sem candidato válido.")
            continue

        # Salva o recorte para inspeção visual
        cv2.imwrite(os.path.join(args.segmented_dir, f"{base_name}_segmented.png"), cropped)

        # Etapa 2 — Pré-processamento da ROI
        preprocessed = preprocess_plate(
            plate_bgr=cropped,
            target_height=120,
            save_debug=args.save_debug,
            debug_prefix=base_name,
            debug_dir=args.debug_dir,
        )

        # Etapa 3 — Tesseract OCR (recebe imagem binarizada: fundo branco, texto preto)
        res = ocr_engine.recognize(image=preprocessed["binary"], image_name=filename)
        results.append(res)

        status = f"{res['plate']} ({res['pattern']})" if res["is_valid"] else f"'{res['cleaned_text']}' [Inválido]"
        print(f"  -> {status:<35} | {res['execution_time']*1000:.1f} ms")

    # Salva resultados em .txt
    output_file = os.path.join(args.results_dir, "resultados_tesseract.txt")
    save_ocr_results(output_file, results, engine_name="Tesseract OCR")

    valid = sum(1 for r in results if r["is_valid"])
    print(f"\n{'='*70}")
    print(f"CONCLUÍDO — {valid}/{total} placas válidas reconhecidas.")
    print(f"Resultado salvo em: {output_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
