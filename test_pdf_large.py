
import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock MilvusClient BEFORE importing standardscraper
mock_milvus = MagicMock()
sys.modules['pymilvus'] = MagicMock()
sys.modules['pymilvus'].MilvusClient = mock_milvus

import pdfprocessor

class TestPDFLargeChunking(unittest.TestCase):
    def test_large_block_splitting(self):
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        # Assume 1 char = 1 token for simplicity in this mock
        mock_tokenizer.encode = lambda text: [0] * len(text)
        
        max_tokens = 100
        
        # Create a text block larger than max_tokens
        # 200 chars > 100 tokens
        large_text = "A" * 200 
        
        # Mock reader
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = large_text
        mock_reader.pages = [mock_page]
        
        chunks = pdfprocessor.chunk_pdf(mock_reader, "http://test.com", mock_tokenizer, max_tokens)
        
        # Verify that we have more than 1 chunk (meaning it was split)
        # Current buggy implementation will return 1 chunk of size 200
        print(f"Chunks returned: {len(chunks)}")
        for i, c in enumerate(chunks):
            print(f"Chunk {i} length: {len(c['chktext'])}")
            
        self.assertTrue(len(chunks) > 1, "Large block was not split!")
        self.assertTrue(len(chunks[0]['chktext']) <= max_tokens, "Chunk 0 exceeded max tokens")

if __name__ == '__main__':
    unittest.main()
