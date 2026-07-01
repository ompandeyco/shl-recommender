import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Intercept and log JSON responses
        async def handle_response(response):
            if "json" in response.headers.get("content-type", ""):
                url = response.url
                if "catalog" in url or "product" in url or "api" in url or "filter" in url or "search" in url:
                    try:
                        data = await response.json()
                        print(f"--- Found JSON API: {url} ---")
                        # print first 500 chars of json to inspect structure
                        print(str(data)[:500])
                    except:
                        pass
        
        page.on("response", handle_response)
        
        print("Navigating to catalog...")
        # Since /solutions/products/product-catalog/ might redirect to /products/, we'll go directly to products search
        await page.goto("https://www.shl.com/products/", wait_until="networkidle")
        
        # wait a bit for any lazy loading
        await asyncio.sleep(5)
        
        print("Done exploring.")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
