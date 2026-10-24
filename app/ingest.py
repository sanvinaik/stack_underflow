import io
import os
import zipfile
import pymupdf as fitz
from pathlib import Path
from dataclasses import dataclass
from typing import List

from app.schemas import Chunk, ChunkMetadata
from app.config import settings

MAX_DRAWINGS_FOR_TABLES = 3000

@dataclass
class ParsedDoc:
    doc_id: str
    pages: list[str]
    chunks: list[Chunk]

def safe_extract(zip_bytes: bytes, dest: Path) -> list[Path]:
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    total_extracted = 0
    extracted_paths = []
    
    dest.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
                
            fname = info.filename
            if fname.startswith('/') or '..' in fname:
                continue
                
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ['.pdf', '.txt']:
                continue
                
            if total_extracted + info.file_size > max_bytes:
                raise ValueError(f"Zip extraction exceeds MAX_UPLOAD_MB ({settings.MAX_UPLOAD_MB}MB)")
                
            total_extracted += info.file_size
            
            safe_name = Path(fname).name
            extracted_path = dest / safe_name
            
            extracted_path.write_bytes(zf.read(info.filename))
            extracted_paths.append(extracted_path)
            
    return extracted_paths

def parse_document(path: Path, doc_id: str) -> ParsedDoc:
    chunks: list[Chunk] = []
    pages: list[str] = []
    filename = path.name
    chunk_idx = 0
    
    if path.suffix.lower() == '.txt':
        text = path.read_text(encoding='utf-8')
        pages.append(text)
        
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) > settings.CHUNK_TARGET_CHARS and current_chunk:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_c{chunk_idx}",
                    content=current_chunk,
                    section_name="Text",
                    metadata=ChunkMetadata(page_numbers=[1], document=filename)
                ))
                chunk_idx += 1
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                
        if current_chunk:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_c{chunk_idx}",
                content=current_chunk,
                section_name="Text",
                metadata=ChunkMetadata(page_numbers=[1], document=filename)
            ))
        return ParsedDoc(doc_id=doc_id, pages=pages, chunks=chunks)

    # PDF Processing
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        print(f"Failed to open {path.name}: {e}")
        return ParsedDoc(doc_id=doc_id, pages=[], chunks=[])

    try:
        for page in doc:
            page_num = page.number + 1
            
            text_blocks = [b for b in page.get_text("blocks") if b[-1] == 0]
            
            # Guarded Table Extraction
            tables = []
            try:
                if len(page.get_cdrawings()) <= MAX_DRAWINGS_FOR_TABLES:
                    tables = page.find_tables(use_layout=False).tables
            except Exception as e:
                print(f"Skipping tables on {path.name} p.{page_num}: {e}")
                tables = []
                
            table_rects = [fitz.Rect(t.bbox) for t in tables]
            
            clean_text_blocks = []
            for b in text_blocks:
                rect = fitz.Rect(b[:4])
                if not any(rect.intersects(tr) for tr in table_rects):
                    clean_text_blocks.append((b[1], "text", b[4].strip()))
                    
            elements = clean_text_blocks
            
            for table in tables:
                bbox = table.bbox
                rows = table.extract()
                md_rows = []
                for idx, row in enumerate(rows):
                    clean_row = [str(c).replace('\n', ' ').strip() if c else "" for c in row]
                    md_rows.append("| " + " | ".join(clean_row) + " |")
                    if idx == 0:
                        md_rows.append("|" + "|".join(["---"] * len(clean_row)) + "|")
                        
                md_table = "\n".join(md_rows)
                elements.append((bbox[1], "table", md_table))
                
            elements.sort(key=lambda x: x[0])
            
            page_text = "\n\n".join(e[2] for e in elements)
            pages.append(page_text)
            
            if not elements:
                continue
                
            current_chunk = ""
            for _, etype, text in elements:
                if len(text) > settings.CHUNK_MAX_CHARS:
                    for i in range(0, len(text), settings.CHUNK_MAX_CHARS):
                        sub_text = text[i:i + settings.CHUNK_MAX_CHARS]
                        chunks.append(Chunk(
                            chunk_id=f"{doc_id}_p{page_num}_c{chunk_idx}",
                            content=sub_text.strip(),
                            section_name=f"Page {page_num}",
                            metadata=ChunkMetadata(page_numbers=[page_num], document=filename)
                        ))
                        chunk_idx += 1
                    continue

                if len(current_chunk) + len(text) > settings.CHUNK_TARGET_CHARS and current_chunk:
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}_p{page_num}_c{chunk_idx}",
                        content=current_chunk.strip(),
                        section_name=f"Page {page_num}",
                        metadata=ChunkMetadata(page_numbers=[page_num], document=filename)
                    ))
                    chunk_idx += 1
                    current_chunk = text
                else:
                    current_chunk += "\n\n" + text if current_chunk else text
                    
            if current_chunk:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_p{page_num}_c{chunk_idx}",
                    content=current_chunk.strip(),
                    section_name=f"Page {page_num}",
                    metadata=ChunkMetadata(page_numbers=[page_num], document=filename)
                ))
                chunk_idx += 1
                
    finally:
        doc.close()
        
    return ParsedDoc(doc_id=doc_id, pages=pages, chunks=chunks)