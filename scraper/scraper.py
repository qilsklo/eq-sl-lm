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

        title = soup.find_all([f'h{x}' for x in range(1,7)])
        text = soup.select('p')
        link = soup.select('a')
        #print(title)
        #[print(T, t, end="\n------\n") for T, t in zip(title, text)]
        #[print(x.get('href')) for x in link]
        print(f"Visited page: {pu}; soup hash: {hash(soup)}")
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


if __name__ == '__main__':
    scrape("https://www.earthquakecountry.org/step3/")