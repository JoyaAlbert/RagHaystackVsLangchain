import os
from typing import List
from PyPDF2 import PdfReader
import pdfplumber

def list_files(directory: str, exts=None) -> List[str]:
    exts = exts or [".pdf", ".txt"]
    files = []
    for root, _, filenames in os.walk(directory):
        for fn in filenames:
            if any(fn.lower().endswith(e) for e in exts):
                files.append(os.path.join(root, fn))
    return files

def read_pdf(path: str) -> str:
    text_parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    except Exception:
        # fallback to pdfplumber which sometimes handles weird encodings
        try:
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    t = p.extract_text()
                    if t:
                        text_parts.append(t)
        except Exception:
            return ""
    return "\n".join(text_parts)

def read_txt(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""

def read_file(path: str) -> str:
    if path.lower().endswith('.pdf'):
        return read_pdf(path)
    else:
        return read_txt(path)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks
