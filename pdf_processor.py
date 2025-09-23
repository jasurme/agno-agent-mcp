#!/usr/bin/env python3
"""
PDF Text Extractor for arXiv Papers
Extracts text content from downloaded PDFs for indexing in OpenSearch.
"""

import os
import json
import PyPDF2
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
import re

def extract_text_with_pypdf2(pdf_path: str) -> str:
    """Extract text using PyPDF2."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"PyPDF2 extraction failed for {pdf_path}: {e}")
        return ""

def extract_text_with_pdfplumber(pdf_path: str) -> str:
    """Extract text using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"pdfplumber extraction failed for {pdf_path}: {e}")
        return ""

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    return text.strip()

def extract_paper_text(pdf_path: str) -> Dict[str, Any]:
    """Extract text from a PDF using multiple methods."""
    print(f"Processing: {Path(pdf_path).name}")
    
    text = extract_text_with_pdfplumber(pdf_path)
    
    if len(text.strip()) < 100:
        print(f"pdfplumber returned minimal text, trying PyPDF2...")
        text = extract_text_with_pypdf2(pdf_path)
    
    cleaned_text = clean_text(text)
    
    chunks = split_text_into_chunks(cleaned_text, chunk_size=1500, overlap=200)
    
    return {
        'full_text': cleaned_text,
        'chunks': chunks,
        'word_count': len(cleaned_text.split()),
        'chunk_count': len(chunks)
    }

def split_text_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for better search results."""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks


def process_all_papers(papers_dir: str, output_dir: str):
    """Process all papers from the papers directory."""
    papers_path = Path(papers_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(papers_path.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found!")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    processed_papers = []
    
    for pdf_file in pdf_files:
        try:
            paper_id = pdf_file.stem
            
            text_data = extract_paper_text(str(pdf_file))
            
            if not text_data['full_text']:
                print(f"No text extracted from {pdf_file.name}")
                continue
            
            paper_data = {
                'paper_id': paper_id,
                'full_text': text_data['full_text'],
                'chunks': text_data['chunks'],
                'word_count': text_data['word_count'],
                'chunk_count': text_data['chunk_count']
            }
            
            output_file = output_path / f"{paper_id}_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, indent=2, ensure_ascii=False)
            
            processed_papers.append(paper_data)
            print(f"Processed: {paper_id}")
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            continue
    
    summary = {
        'total_papers': len(processed_papers),
        'total_words': sum(paper['word_count'] for paper in processed_papers),
        'total_chunks': sum(paper['chunk_count'] for paper in processed_papers),
        'papers': [
            {
                'paper_id': paper['paper_id'],
                'word_count': paper['word_count'],
                'chunk_count': paper['chunk_count']
            }
            for paper in processed_papers
        ]
    }
    
    summary_file = output_path / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Processing Summary ===")
    print(f"Papers processed: {summary['total_papers']}")
    print(f"Total words: {summary['total_words']:,}")
    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Output directory: {output_path}")

def main():
    """Main function to process all papers."""
    print("=== PDF Text Extractor ===")
    
    papers_dir = "papers/pdfs"
    output_dir = "papers/processed"
    
    if not Path(papers_dir).exists():
        print(f"Papers directory not found: {papers_dir}")
        return
    
    process_all_papers(papers_dir, output_dir)

if __name__ == "__main__":
    main()
