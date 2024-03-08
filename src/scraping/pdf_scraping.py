#%%
from urllib.parse import urljoin
from requests_html import HTMLSession
from tqdm import tqdm
from fake_useragent import UserAgent
from datetime import datetime
from src.scraping.models.pdf_model import PDFLinkModel
from src.scraping.adapters.tsl_adapter import TLSAdapter
import hashlib
from bs4 import BeautifulSoup

class PDFScraping:
    def __init__(self, name, host, selector, depth, verify=True) -> None:
        self.name = name
        self.host = host
        self.selector = selector
        self.depth_search = depth
        self.verify = verify
        self.user_agent = UserAgent() 
        self.session = HTMLSession()
        self.session.mount("https://", TLSAdapter())

    def is_pdf_link(self, url):
        """
        Check if a given URL points to a PDF file.

        Args:
            url (str): The URL to be checked.

        Returns:
            bool: True if the URL points to a PDF file, False otherwise.

        Note:
            This method first checks if the URL ends with '.pdf'. If it does, it returns True immediately.
            If not, it sends a GET request to the URL and checks the 'Content-Type' header in the response.
            If the header indicates that the content is a PDF ('application/pdf'), it returns True.
            If the above checks fail, it reads the beginning of the file content and checks if it starts with '%PDF'.
            If it does, it returns True. Otherwise, it returns False.

        Example:
            >>> example_url = 'https://example.com/example.pdf'
            >>> is_pdf = is_pdf_link(self, example_url)
            >>> print(is_pdf)
            True
        """
        try:
            if url.lower().endswith('.pdf'):
                return True
            
            response = self.session.get(url, headers={'Range': 'bytes=0-3'}, timeout=5, verify=self.verify)
            
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' in content_type:
                return True
            
            file_content = response.content
            if file_content.startswith(b'%PDF'):
                return True

            return False

        except Exception:
            # print(f'Erro pdf link: {e}')
            return False
        
    def get_headers(self):
        """
        Generate HTTP headers with a random User-Agent.

        Returns:
            dict: A dictionary containing HTTP headers with a random User-Agent.

        Note:
            This method generates HTTP headers with a random User-Agent using the 'user_agent' attribute of the class instance.

        Example:
            >>> headers = get_headers(self)
            >>> print(headers)
            {'User-Agent': '<random user-agent>'}
        """
        return {'User-Agent': self.user_agent.random}

    def get_links_recursive(self, url, depth, visited_urls=set(), selector_css=None):
        """
        Recursively retrieve links from a given URL up to a specified depth.

        Args:
            url (str): The URL from which to start retrieving links.
            depth (int): The maximum depth of recursion.
            visited_urls (set, optional): A set of visited URLs to avoid revisiting. Defaults to an empty set ().
            selector_css (str, optional): CSS selector to filter the links. Defaults to None.

        Returns:
            set: A set containing unique absolute links retrieved recursively.

        Note:
            This method retrieves links from a given URL up to the specified depth recursively.
            It avoids revisiting URLs already visited to prevent infinite recursion.
            If a CSS selector is provided, it filters the links based on the selector.
            The recursion depth is controlled by the 'depth' parameter.

        Example:
            >>> example_url = 'https://example.com'
            >>> depth = 2
            >>> selector_css = 'a[href^="https://"]'
            >>> links = get_links_recursive(self, example_url, depth, selector_css=selector_css)
            >>> print(links)
            {'https://example.com/link1', 'https://example.com/link2', ...}
        """
        if depth == 0 or url in visited_urls:
            return []

        visited_urls.add(url)

        try:
            with self.session.get(url, headers=self.get_headers(), verify=self.verify) as response:
                links = []
                if selector_css:
                    body_selected = response.html.find(selector_css, first=True)
                    if body_selected:
                        links = body_selected.links
                else:
                    links = response.html.find('body', first=True).links

                absolute_links = [urljoin(url, link) for link in links]
                print(f"Links depth {depth} from {url}:\n{absolute_links}\n")

                recursive_links = []
                for link in absolute_links:
                    recursive_links.extend(self.get_links_recursive(link, depth - 1, visited_urls, selector_css))

                return set(absolute_links + recursive_links)

        except Exception:
            # print(f'Error {e} url: {url}')
            return []
        
    def get_pdfs_links(self):
        """
        Scrape the web for PDF links.

        Prints the host URL being scraped, retrieves links recursively, filters PDF links,
        and creates PDFLinkModel instances for each PDF link found.

        Returns:
            list: A list of PDFLinkModel instances representing PDF links found on the website.

        Note:
            This method performs web scraping to find PDF links on the specified host URL.
            It uses the `get_links_recursive` method to recursively retrieve links from the host.
            If a CSS selector is provided, it filters the links based on the selector.
            It then filters the retrieved links to find PDF links using the `is_pdf_link` method.
            Finally, it creates instances of PDFLinkModel for each PDF link found.

        Example:
            >>> pdf_scraper = PDFScraper(...)
            >>> pdf_links = pdf_scraper.get_pdfs_links()
            >>> for link in pdf_links:
            >>>     print(link.url)
            'https://example.com/example.pdf'
        """
        print(f'Web scraping url: {self.host}')
        links = self.get_links_recursive(
            url=self.host,
            selector_css=self.selector if self.selector != '' else None,
            depth=self.depth_search
        )

        print(f'Getting pdfs links...')
        pdfs = [PDFLinkModel(link, self.name, datetime.now()) for link in tqdm(set(links)) if self.is_pdf_link(link)]

        return pdfs
    
    def create_hash(self, parser = 'html.parser'):
        """
        Create a SHA-256 hash of the content of a web page.

        Args:
            parser (str, optional): The parser to use for parsing the web page content. Defaults to 'html.parser'.

        Returns:
            str or None: The SHA-256 hash of the web page content if successful, None otherwise.

        Note:
            This method sends a GET request to the host URL, retrieves the page content,
            parses it using BeautifulSoup with the specified parser,
            generates a SHA-256 hash of the page content, and returns the hash.

        Example:
            >>> url = 'https://example.com'
            >>> parser = 'html.parser'
            >>> pdf_scraper = PDFScraper(...)
            >>> page_hash = pdf_scraper.create_hash(parser)
            >>> print(page_hash)
            '5a8f7ab55a7961e568640bd1437d6e55b033dfada68d727d1d1670c382c121f'
        """
        try:
            with self.session.get(self.host, headers=self.get_headers(), verify=self.verify) as response:
                soup = BeautifulSoup(response.content, parser)
                page_content = soup.get_text()
                hash = hashlib.sha256(page_content.encode()).hexdigest()
                return hash
        except Exception:
            return None