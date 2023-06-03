import requests

def read_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None
url = "https://en.wikipedia.org/wiki/Pencil"
from bs4 import BeautifulSoup


html_code = read_website(url)
soup = BeautifulSoup(html_code, 'html.parser')
plain_text = soup.get_text()

print(plain_text)