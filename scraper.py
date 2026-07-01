import asyncio
import json
import time
from pathlib import Path
from playwright.async_api import async_playwright

async def run_scraper():
    print("Starting Playwright scraper...")
    
    # Path to save the catalog
    out_path = Path("data/catalog.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # The URL specified by the user
        url = "https://www.shl.com/solutions/products/product-catalog/"
        print(f"Navigating to {url}...")
        
        try:
            await page.goto(url, wait_until="networkidle")
        except Exception as e:
            print(f"Error navigating to page: {e}")
            await browser.close()
            return

        # Handle cookie banners if any
        try:
            # SHL usually uses a generic Accept Cookies button if in EU/UK. Just a quick attempt.
            await page.click("text=Accept All Cookies", timeout=3000)
        except:
            pass
            
        print("Waiting for page load and looking for 'Individual Test Solutions' filter...")

        try:
            # Attempt to filter by "Individual Test Solutions"
            # This could be a checkbox, a dropdown, or a tab.
            filter_locator = page.locator("text='Individual Test Solutions'")
            if await filter_locator.count() > 0:
                # If it's a checkbox or label, click it.
                await filter_locator.first.click()
                print("Clicked 'Individual Test Solutions' filter.")
                await asyncio.sleep(2)  # wait for results to filter
        except Exception as e:
            print(f"Could not find or click the 'Individual Test Solutions' filter: {e}")

        # Now we iterate through pagination
        has_next_page = True
        page_num = 1

        while has_next_page:
            print(f"Scraping page {page_num}...")
            # Wait for list to render (adjust selector based on actual DOM)
            try:
                await page.wait_for_selector(".product-item, .assessment-card, tr.catalog-row", timeout=5000)
            except:
                print("No product elements found on this page. DOM structure might be different.")
                pass
            
            # Extract elements (broad selectors trying to catch tables or lists)
            elements = await page.locator(".product-item, .assessment-card, tr.catalog-row").all()
            if not elements:
                # Fallback generic locator if the specific classes aren't present
                elements = await page.locator("article, .catalog-list li").all()

            for el in elements:
                try:
                    # Name and URL usually in an anchor tag
                    title_el = el.locator("h2, h3, .product-title, .title a").first
                    name = await title_el.inner_text() if await title_el.count() > 0 else "Unknown"
                    
                    # Try finding URL in an anchor tag within the element
                    a_tag = el.locator("a").first
                    href = await a_tag.get_attribute("href") if await a_tag.count() > 0 else ""
                    
                    if href and not href.startswith("http"):
                        href = f"https://www.shl.com{href}"

                    # Description
                    desc_el = el.locator(".description, p").first
                    description = await desc_el.inner_text() if await desc_el.count() > 0 else None

                    # Test Type (letter code like K, P, A, B, S)
                    # Often represented as an icon or specific badge class
                    type_el = el.locator(".test-type, .badge, .letter-code").first
                    test_type = await type_el.inner_text() if await type_el.count() > 0 else None

                    # Remote testing / adaptive attributes (look for checkmarks or labels)
                    text_content = await el.inner_text()
                    text_lower = text_content.lower()
                    remote_testing = "remote testing" in text_lower or "remote testing: yes" in text_lower
                    adaptive = "adaptive" in text_lower or "irt" in text_lower

                    # Duration
                    duration = None
                    if "mins" in text_lower or "minutes" in text_lower:
                        # naive extraction
                        import re
                        match = re.search(r'(\d+)\s*(?:mins|minutes)', text_lower)
                        if match:
                            duration = int(match.group(1))

                    if name != "Unknown":
                        catalog.append({
                            "name": name.strip(),
                            "url": href,
                            "test_type": test_type.strip() if test_type else None,
                            "description": description.strip() if description else None,
                            "remote_testing": remote_testing,
                            "adaptive_irt": adaptive,
                            "duration_minutes": duration
                        })
                except Exception as ex:
                    # Fallback to avoid crashing
                    print(f"Skipping an element due to error: {ex}")
                    continue

            # Pagination
            try:
                next_btn = page.locator("text='Next', .pagination-next, [aria-label='Next page']")
                if await next_btn.count() > 0 and await next_btn.first.is_visible() and not await next_btn.first.is_disabled():
                    print("Found 'Next' button, clicking...")
                    await next_btn.first.click()
                    page_num += 1
                    time.sleep(2)  # 2 second delay to be polite
                else:
                    has_next_page = False
            except Exception as e:
                print("Pagination finished or not found.")
                has_next_page = False

        await browser.close()

    # Save to JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(f"\n--- Scraping Complete ---")
    print(f"Total assessments collected: {len(catalog)}")
    
    if catalog:
        print("\nSample entries:")
        print(json.dumps(catalog[:3], indent=2))

if __name__ == "__main__":
    asyncio.run(run_scraper())
