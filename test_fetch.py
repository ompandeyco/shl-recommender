import urllib.request
import re

url = 'https://www.shl.com/solutions/products/product-catalog/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req).read().decode('utf-8')
    with open("shl_catalog_raw.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Fetched successfully. Length:", len(html))
except Exception as e:
    print("Error:", e)
