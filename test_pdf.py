
import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock MilvusClient BEFORE importing standardscraper
mock_milvus = MagicMock()
sys.modules['pymilvus'] = MagicMock()
sys.modules['pymilvus'].MilvusClient = mock_milvus

import io
import pypdf
# Now import standardscraper, which will use the mocked MilvusClient
import standardscraper
import pdfprocessor

class TestPDFScraping(unittest.TestCase):
    def test_pdf_processing(self):
        mock_response = MagicMock()
        mock_response.url = "http://example.com/test.pdf"
        mock_response.content = b"%PDF-1.4..." 
        
        with patch('pypdf.PdfReader') as MockReader:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "This is a test PDF content.\nIt has multiple lines."
            
            instance = MockReader.return_value
            instance.pages = [mock_page]
            
            # Mock db_store
            with patch('standardscraper.db_store') as mock_db_store:
                standardscraper.process_scrape_pdf(mock_response)
                
                mock_db_store.assert_called_once()
                chunks = mock_db_store.call_args[0][0]
                self.assertTrue(len(chunks) > 0)
                self.assertIn("This is a test PDF content", chunks[0]['chktext'])
                self.assertEqual(chunks[0]['origin'], "http://example.com/test.pdf")
                self.assertEqual(chunks[0]['heading'], "PDF Content")

if __name__ == '__main__':
    unittest.main()
