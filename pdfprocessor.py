import pypdf
import re

def estimate_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def chunk_pdf(reader, url, tokenizer, max_tokens, title="", author=""):
    """
    Extracts text from a pypdf.PdfReader object, handles list formatting,
    and chunks text into token-limited segments preserving semantic context.
    Returns chunks matching the myshake_pdf schema.
    """
    
    all_chunks = []
    
    # Process page by page to keep track of page numbers
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text:
            continue
            
        # Basic cleaning
        bullet_patterns = ['•', '●', '○', '■', '▪']
        for bullet in bullet_patterns:
            page_text = page_text.replace(bullet, '-')

        # Split into paragraphs
        raw_paragraphs = re.split(r'\n\s*\n', page_text)
        
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
        
        # Chunk the blocks for this page
        page_chunks = []
        current_chunk_text = ""
        current_chunk_tokens = 0
        
        for block in semantic_blocks:
            block_token_count = estimate_tokens(block, tokenizer)
            
            if block_token_count > max_tokens:
                # If the block itself is too big, split it
                if current_chunk_text:
                    page_chunks.append(current_chunk_text.strip())
                    current_chunk_text = ""
                    current_chunk_tokens = 0
                
                sub_blocks = recursive_split(block, tokenizer, max_tokens)
                for sub in sub_blocks:
                    page_chunks.append(sub.strip())
                continue

            if current_chunk_tokens + block_token_count > max_tokens:
                if current_chunk_text:
                    page_chunks.append(current_chunk_text.strip())
                
                current_chunk_text = block
                current_chunk_tokens = block_token_count
            else:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + block
                else:
                    current_chunk_text = block
                
                current_chunk_tokens += block_token_count

        if current_chunk_text:
            page_chunks.append(current_chunk_text.strip())
            
        # Create chunk objects for this page
        for c in page_chunks:
            all_chunks.append({
                "text": c,
                "title": title,
                "page_num": page_num,
                "author": author,
                "doi": "", # Placeholder, could be extracted if we had a robust way
                "publication_year": 0 # Placeholder
            })

    return all_chunks

def recursive_split(text, tokenizer, max_tokens):
    token_count = estimate_tokens(text, tokenizer)
    if token_count <= max_tokens:
        return [text]
    
    # 1. Try splitting by sentence
    parts = re.split(r'(?<=[.!?])\s+', text)
    separator = " " # Join with space when rebuilding
    
    if len(parts) == 1:
        # 2. Try splitting by space (words)
        parts = text.split(' ')
        separator = " "
        
        if len(parts) == 1:
            # 3. Hard split by char
            # Calculate safe char limit based on actual token density
            avg_chars_per_token = len(text) / max(1, token_count)
            # Use 90% of capacity to be safe
            chunk_chars = int(max_tokens * avg_chars_per_token * 0.9)
            chunk_chars = max(1, chunk_chars)
            return [text[i:i+chunk_chars] for i in range(0, len(text), chunk_chars)]
    
    # Combine parts
    chunks = []
    current = ""
    for part in parts:
        # For sentence split, re.split consumes the whitespace, so we add it back
        # For word split, we need a space
        prefix = separator if current else ""
        candidate = current + prefix + part
        
        if estimate_tokens(candidate, tokenizer) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part
            # If the single part is still too big, recurse
            if estimate_tokens(current, tokenizer) > max_tokens:
                sub = recursive_split(current, tokenizer, max_tokens)
                chunks.extend(sub[:-1])
                current = sub[-1]
    
    if current:
        chunks.append(current)
    return chunks

