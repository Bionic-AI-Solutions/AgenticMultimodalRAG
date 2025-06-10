import pytest
from app.main import detect_file_type, extract_text, extract_pdf, extract_image, extract_audio, extract_content_by_type, chunk_text_recursive

class TestExtraction:
    def test_detect_file_type_text(self):
        content = b'Hello world!'
        assert detect_file_type('test.txt', content).startswith('text/')

    def test_extract_text_utf8(self):
        content = 'Hello world!'.encode('utf-8')
        assert extract_text(content) == 'Hello world!'

    def test_extract_text_latin1(self):
        content = 'Olá mundo!'.encode('latin1')
        assert 'Olá' in extract_text(content)

    def test_extract_pdf_stub(self):
        content = b'%PDF-1.4...'
        assert 'PDF extraction' in extract_pdf(content)

    def test_extract_image_stub(self):
        content = b'\x89PNG...'
        assert 'OCR' in extract_image(content)

    def test_extract_audio_stub(self):
        content = b'ID3...'
        assert 'ASR' in extract_audio(content)

    def test_extract_content_by_type_text(self):
        content = b'Hello world!'
        assert extract_content_by_type('text/plain', content) == 'Hello world!'

    def test_chunk_text_recursive(self):
        text = ' '.join(['word']*1000)
        chunks = chunk_text_recursive(text, chunk_size=100, overlap=20)
        assert all(len(chunk.split()) <= 100 for chunk in chunks)
        assert len(chunks) > 1 