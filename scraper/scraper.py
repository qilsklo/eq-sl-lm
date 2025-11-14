"""
to scrape
https://www.earthquakecountry.org/
https://www.usgs.gov/programs/earthquake-hazards/faqs-category
maybe go one level deep into links on this site
- all headings
- all paragraphs
- all lists (ul, ol) and their list items (li)

"""


import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse


def scrape(pageurl: str):
    memo = set()
    def scr(pu):
        if not is_valid_scheme(pu): 
            #print(f"BAD URL: {pu}")
            return None
        memo.add(pu)
        response = requests.get(pu)
        soup = BeautifulSoup(response.text, 'html.parser')
        link = soup.select('a')
        process_scrape(soup)
        if extract_domain(pu) == extract_domain(pageurl): #Recur
            for x in link:
                y = x.get('href')
                if y not in memo:
                    scr(y)
    scr(pageurl)

def extract_domain(url):
    return urlparse(url).netloc
def is_valid_scheme(url):
    x = urlparse(url).scheme
    return "http" in x
def process_scrape(soup):
    
    global elements, chunks # @dev
    elements = soup.select("h1, h2, h3, h4, h5, h6, p, li")
    chunks = chunk_soup(soup)
    

    print(f"Visited page; soup hash: {hash(soup)}")

def chunk_soup(soup):
    """
    Inside a vector DB entry:
    - embedding
    - origin URL
    - title of heading of the chunk
    - original HTML Snippet of the chunk
    1. Take the soup, and create a dict {url, heading, html_snippet}
    2. for each chunk create a dictionary that includes all this info 
    """
    
    chunks = []
    current_chunk = []

    for el in elements:
        text = el.get_text(" ", strip=True)

        if el.name.startswith("h"):
            # start a new section
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            current_chunk.append(text)

        elif el.name == "li":
            current_chunk.append(f"- {text}")

        else:
            current_chunk.append(text)

    # add last chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    [print(c, end="\n\n") for c in chunks]

    # turn `chunks` into an array of dictionaries 

    return chunks

def process_chunks(soup, chunks):

    # store in vectordb
    # store same chunks in a bm25 index
    ...

def vectorize(string):
    ...

def answer_query(query):
    ...

if __name__ == '__main__':
    scrape("https://www.earthquakecountry.org/step3/")