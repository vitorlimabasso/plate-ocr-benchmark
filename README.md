# ALPR OCR Comparison — Segmentação de Placas e Comparação Tesseract vs. EasyOCR

> **Atividade Acadêmica de Processamento Digital de Imagens (PDI)**  
> Pipeline completo de Visão Computacional para detecção, segmentação por PDI clássico e reconhecimento óptico de caracteres em placas veiculares brasileiras (Mercosul e Antiga/Cinza).

---

## 📁 Estrutura do Repositório

O projeto está organizado em três pastas principais independentes:

```text
├── resultados txt/
│   ├── resultados_tesseract.txt    # Saída estruturada do Tesseract OCR (acertos, padrões, tempos)
│   ├── resultados_easyocr.txt      # Saída estruturada do EasyOCR
│   └── relatorio_comparativo.txt   # Tabela comparativa consolidada, métricas e análise técnica
│
├── pipeline TesseractOCR/
│   ├── main.py                     # Script principal de execução do pipeline Tesseract
│   ├── requirements.txt            # Dependências específicas para este pipeline
│   ├── data/
│   │   ├── raw/                    # Diretório para inserir as imagens de entrada
│   │   ├── segmented/              # Placas recortadas salvas automaticamente
│   │   └── debug_steps/            # Etapas intermediárias de PDI para validação visual
│   ├── results/                    # Resultados gerados em formato .txt
│   └── src/
│       ├── segmentation.py         # Segmentação por PDI clássico (Sobel + Canny + Morfologia)
│       ├── preprocessing.py        # Pré-processamento (Bilateral, CLAHE, Binarização de Otsu)
│       ├── ocr_tesseract.py        # Motor Tesseract com fallback adaptativo de PSM
│       └── utils.py                # Regex de placas, correções heurísticas e métricas
│
└── pipeline EasyOCR/
    ├── main.py                     # Script principal de execução do pipeline EasyOCR
    ├── requirements.txt            # Dependências específicas para este pipeline (PyTorch + EasyOCR)
    ├── data/
    │   ├── raw/                    # Diretório para inserir as imagens de entrada
    │   ├── segmented/              # Placas recortadas salvas automaticamente
    │   └── debug_steps/            # Etapas intermediárias de PDI para validação visual
    ├── results/                    # Resultados gerados em formato .txt
    └── src/
        ├── segmentation.py         # Segmentação por PDI clássico
        ├── preprocessing.py        # Pré-processamento (LAB-CLAHE para realce de caracteres)
        ├── ocr_easyocr.py          # Motor EasyOCR (CRAFT + CRNN/ResNet)
        └── utils.py                # Regex de placas, correções heurísticas e métricas
```

---

## 🔬 Pipeline de Processamento Digital de Imagens (PDI)

Ambos os pipelines utilizam a **exata mesma metodologia clássica de PDI** para detecção e segmentação (sem redes neurais de detecção):

```
Imagem de Entrada (data/raw/)
        │
        ▼
[1] Conversão em Tons de Cinza + Filtro Bilateral (atenuação de ruído com preservação de bordas)
        │
        ▼
[2] Extração de Gradientes:
    ├── Estágio A: Gradiente Horizontal de Sobel (dx=1, dy=0) + Binarização de Otsu
    └── Estágio B: Detecção de Bordas via Canny com limiares adaptativos
        │
        ▼
[3] Fechamento Morfológico Multi-Escala (kernels retangulares horizontais alargados: 45x9, 35x9, 25x7)
        │
        ▼
[4] Filtragem e Scoring Multiobjetivo dos Contornos:
    └── Score = (Aspect Ratio ~ 4.2) + (Energia Sobel interna) + (Desvio Padrão) + (Solidez)
        │
        ▼
[5] Recorte da Placa (ROI) com margem de segurança (padding)
        │
        ▼
[6] Pré-processamento Otimizado para OCR:
    ├── Tesseract: Redimensionamento Bicúbico + CLAHE + Binarização de Otsu + Inversão de Polaridade
    └── EasyOCR:   Redimensionamento Bicúbico + Realce de Contraste no canal L (espaço de cores LAB)
        │
        ▼
[7] Reconhecimento Óptico de Caracteres & Pós-processamento com Regex e Heurísticas
```

---

## 🚀 Como Executar

### 1. Pré-requisitos Gerais
- **Python 3.8+**
- Para o **Tesseract OCR**, certifique-se de ter o executável instalado no sistema:
  - **Windows:** Instale pelo [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) (padrão em `C:\Program Files\Tesseract-OCR\tesseract.exe`).
  - **Linux:** `sudo apt-get install tesseract-ocr`.

---

### 2. Executando o Pipeline Tesseract OCR

```bash
cd "pipeline TesseractOCR"

# Instalar dependências
pip install -r requirements.txt

# Inserir as imagens a serem processadas na pasta data/raw/
# Em seguida, executar:
python main.py
```

Os resultados e relatórios serão gravados em `pipeline TesseractOCR/results/resultados_tesseract.txt`.

---

### 3. Executando o Pipeline EasyOCR

```bash
cd "pipeline EasyOCR"

# Instalar dependências
pip install -r requirements.txt

# Inserir as imagens a serem processadas na pasta data/raw/
# Em seguida, executar:
python main.py

# (Opcional) Para forçar a execução via CPU:
python main.py --cpu-only
```

*Nota: Na primeira execução, o EasyOCR fará o download automático dos modelos pré-treinados (`CRAFT` e `CRNN`).*

---

## 📊 Resultados e Análise Comparativa

Testes realizados sobre o dataset de 20 amostras de placas brasileiras em condições reais de iluminação e perspectiva:

| Métrica | Tesseract OCR | EasyOCR |
| :--- | :---: | :---: |
| **Total de Imagens** | 20 | 20 |
| **Placas Válidas Reconhecidas** | **14 (70.0%)** | **8 (40.0%)** |
| **Padrão Mercosul** | 13 | 7 |
| **Padrão Antigo / Cinza** | 1 | 1 |
| **Tempo Médio por Imagem** | ~298 ms | ~250 ms |

### Conclusões Técnicas:
1. **Impacto do Pré-processamento:** A cadeia de PDI clássico (CLAHE + Otsu + normalização de polaridade) foi fundamental para que o Tesseract alcançasse 70% de assertividade sobre as placas binarizadas.
2. **Tesseract OCR:** Demonstrou maior precisão quando os caracteres estão bem binarizados e separados do fundo, beneficiando-se do fallback adaptativo de PSM (Page Segmentation Mode 7/6/8).
3. **EasyOCR:** Baseado em Deep Learning (CRAFT + ResNet), tem excelente robustez contra variações geométricas e rotações, porém pode sofrer com fragmentação de caracteres em fontes específicas quando há reflexos intensos na chapa metálica.
