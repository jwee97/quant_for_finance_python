"""Optical character recognition for scanned filings, through an engine that is installed (PaddleOCR or Surya), never one that is assumed.

A scanned report has no text layer, and its digits are pixels. An OCR engine turns them back into words with positions, which the PDF reader's row logic then reads like any other page; the result is marked
``pdf-ocr`` and always needs a person's review, because a misread digit is a wrong number that looks like a right one. The engines are heavy and optional (PaddleOCR needs PaddlePaddle; Surya needs PyTorch and
downloads its models on first use), so nothing here imports them until asked, and a missing engine is named in the message with how to install it. Languages: Japanese, Korean, Chinese and English.

The adapters were written against the engines' documented interfaces and are tested here with a stand-in engine; they have not been run against the real engines in this environment.
"""

from __future__ import annotations

import importlib.util

from ..errors import FundamentalsError

LANG = {"en": {"paddleocr": "en", "surya": ["en"]}, "ja": {"paddleocr": "japan", "surya": ["ja"]}, "ko": {"paddleocr": "korean", "surya": ["ko"]}, "zh": {"paddleocr": "ch", "surya": ["zh"]},
        "zh-tw": {"paddleocr": "chinese_cht", "surya": ["zh"]}}
ENGINES = ("rapidocr", "paddleocr", "surya")                      # lightest first: RapidOCR runs on onnxruntime with models bundled in the wheel (no model download, no PaddlePaddle or PyTorch)
MODULES = {"rapidocr": "rapidocr_onnxruntime", "paddleocr": "paddleocr", "surya": "surya"}


def available() -> list[str]:
    return [e for e in ENGINES if importlib.util.find_spec(MODULES[e]) is not None]


def missing_message() -> str:
    return "this PDF has no text layer (it looks scanned) and no OCR engine is installed: install RapidOCR (pip install rapidocr-onnxruntime: small, runs on onnxruntime, English and Chinese models bundled), PaddleOCR (pip install paddleocr paddlepaddle) or Surya (pip install surya-ocr), or use the company's XBRL or an Excel/CSV export"


RENDER_DPI = 200


def render_pages(data: bytes, dpi: int = RENDER_DPI, limit: int = 60) -> list[bytes]:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    return [page.get_pixmap(dpi=dpi).tobytes("png") for page in list(doc)[:limit]]


class Engine:
    name = "engine"

    def words(self, image_png: bytes, language: str) -> list[tuple]:
        """``[(x0, y_mid, x1, text, confidence)]`` for one page image."""
        raise NotImplementedError


def split_line(x0: float, y: float, x1: float, text: str) -> list[tuple]:
    """A recognised *line* of text (what most OCR engines return) as words with boxes: the line's width shared out by characters, so the layout reader can tell a label from the figures after it."""
    words = str(text).split()
    if len(words) <= 1:
        return [(x0, y, x1, str(text).strip())] if str(text).strip() else []
    total = sum(len(w) for w in words) + (len(words) - 1)
    per = (x1 - x0) / max(total, 1)
    out, cur = [], x0
    for w in words:
        out.append((cur, y, cur + per * len(w), w))
        cur += per * (len(w) + 1)
    return out


class RapidEngine(Engine):
    """RapidOCR on onnxruntime: PP-OCR detection and recognition models that ship inside the wheel. English and Chinese are read natively; Japanese and Korean pages are read only as far as the Chinese
    model's character set goes, which is not enough for their kana and hangul: use PaddleOCR or Surya for those."""
    name = "rapidocr"

    def __init__(self):
        self._ocr = None

    def words(self, image_png: bytes, language: str) -> list[tuple]:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise FundamentalsError(missing_message()) from None
        if self._ocr is None:
            self._ocr = RapidOCR()
        result, _ = self._ocr(image_png)
        out = []
        for box, text, score in (result or []):
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            for x0, y, x1, t in split_line(min(xs), (min(ys) + max(ys)) / 2, max(xs), text):
                out.append((x0, y, x1, t, float(score)))
        return out


class PaddleEngine(Engine):
    name = "paddleocr"

    def __init__(self):
        self._by_lang: dict = {}

    def words(self, image_png: bytes, language: str) -> list[tuple]:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise FundamentalsError(missing_message()) from None
        lang = LANG.get(language, LANG["en"])["paddleocr"]
        if lang not in self._by_lang:
            self._by_lang[lang] = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        out = []
        result = self._by_lang[lang].ocr(image_png, cls=True) or []
        for page in result:
            for box, (text, conf) in (page or []):
                xs, ys = [p[0] for p in box], [p[1] for p in box]
                out.append((min(xs), (min(ys) + max(ys)) / 2, max(xs), str(text), float(conf)))
        return out


class SuryaEngine(Engine):
    name = "surya"

    def words(self, image_png: bytes, language: str) -> list[tuple]:
        try:
            import io

            from PIL import Image
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
        except ImportError:
            raise FundamentalsError(missing_message()) from None
        image = Image.open(io.BytesIO(image_png)).convert("RGB")
        predictions = RecognitionPredictor()([image], [LANG.get(language, LANG["en"])["surya"]], DetectionPredictor())
        out = []
        for line in predictions[0].text_lines:
            x0, y0, x1, y1 = line.bbox
            out.append((x0, (y0 + y1) / 2, x1, line.text, float(getattr(line, "confidence", 0.0) or 0.0)))
        return out


def engine(name: str | None = None) -> Engine:
    """The engine asked for, or the first installed one."""
    choice = name or next(iter(available()), None)
    if choice is None:
        raise FundamentalsError(missing_message())
    if choice not in ENGINES:
        raise FundamentalsError(f"OCR engine must be one of {', '.join(ENGINES)}")
    return {"rapidocr": RapidEngine, "paddleocr": PaddleEngine, "surya": SuryaEngine}[choice]()


def pages_subset(data: bytes, language: str, eng: Engine | None, wanted: list[int]) -> dict[int, list[tuple]]:
    """OCR of only the 1-based pages asked for (the targeted fallback never reads a whole report again for a few bad pages)."""
    eng = eng or engine()
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    out = {}
    for n in wanted:
        if 1 <= n <= len(doc):
            image = doc[n - 1].get_pixmap(dpi=RENDER_DPI).tobytes("png")
            eng.page_hint = n                                                          # real engines ignore it; a test double that reads from a known document uses it
            out[n] = [(w[0], w[1], w[2], w[3]) for w in eng.words(image, language)]
    return out


def pages(data: bytes, language: str = "en", eng: Engine | None = None) -> list[list[tuple]]:
    """The words of every page of a scanned PDF, as ``(x0, y_mid, x1, text)`` with a ``conf`` kept aside in :data:`LAST_CONFIDENCE`."""
    eng = eng or engine()
    out = []
    LAST_CONFIDENCE.clear()
    for n, image in enumerate(render_pages(data), start=1):
        words = eng.words(image, language)
        out.append([(w[0], w[1], w[2], w[3]) for w in words])
        for w in words:
            LAST_CONFIDENCE[(n, w[3])] = w[4]
    return out


LAST_CONFIDENCE: dict = {}
