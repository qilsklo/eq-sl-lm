import scraper
import pypdf
import re

tokenizer = scraper.embedding_fn.tokenizer
max_tokens = scraper.max_tokens

def estimate_tokens(text):
    """
    Estimates token count for sentence-transformers/all-MiniLM-L6-v2.
    Roughly 1 word ~= 1.3 tokens. 
    We use a slightly conservative multiplier to ensure we stay under 256.
    """
    return len(tokenizer.encode(text))

def chunk_pdf(reader, url):
    """
    Extracts text from a pypdf.PdfReader object, handles list formatting,
    and chunks text into token-limited segments preserving semantic context.
    """
    
    full_text = ""
    
    # 1. Extract text from all pages
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"


    bullet_patterns = ['•', '●', '○', '■', '▪']
    for bullet in bullet_patterns:
        full_text = full_text.replace(bullet, '-')

    raw_paragraphs = re.split(r'\n\s*\n', full_text)
  
    semantic_blocks = []
    if raw_paragraphs:
        current_block = raw_paragraphs[0].strip()
        
        for i in range(1, len(raw_paragraphs)):
            next_para = raw_paragraphs[i].strip()
            if not next_para:
                continue
                
            is_list_item = next_para.strip().startswith('-') or next_para.strip().startswith('*')
            
            if is_list_item:
                current_block += "\n" + next_para
            else:
                semantic_blocks.append(current_block)
                current_block = next_para
        
        if current_block:
            semantic_blocks.append(current_block)

    final_chunks = []
    current_chunk_text = ""
    current_chunk_tokens = 0
    
    for block in semantic_blocks:
        block_token_count = estimate_tokens(block)
        
        if current_chunk_tokens + block_token_count > max_tokens:
            if current_chunk_text:
                final_chunks.append(current_chunk_text.strip())
            
            current_chunk_text = block
            current_chunk_tokens = block_token_count
        else:
            if current_chunk_text:
                current_chunk_text += "\n\n" + block
            else:
                current_chunk_text = block
            
            current_chunk_tokens += block_token_count

    if current_chunk_text:
        final_chunks.append(current_chunk_text.strip())
    f = [{"chktext":c, "origin":url, "heading":"","html_snippet":""} for c in final_chunks]
    print(f[0])
    return f