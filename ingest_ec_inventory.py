"""Try to fetch the full EC inventory export published by ECHA.

ECHA's EC inventory web page only exports the first 20,000 rows of a
search. A full CSV has been published under the ECHA documents library
(URL below, referenced in ECHA's answer to an access-to-documents request
and in third-party tooling). The documents library sits behind the same
Azure WAF as the list pages, so a headed Chrome is used.

If the file cannot be retrieved, a gap is recorded in raw/fetch_meta.json
and nothing is approximated. The pipeline does not depend on this file:
EC and CAS numbers come from the lists themselves, and structures from PubChem.
"""
import asyncio, datetime, json, pathlib
from playwright.async_api import async_playwright

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
URL = "https://echa.europa.eu/documents/10162/17222/ec_inventory_en.csv/326d9adb-27ed-5460-a2da-4f651b81e4b3"


async def main():
    RAW.mkdir(exist_ok=True)
    meta_path = RAW / "fetch_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rec = {"source_page": "https://echa.europa.eu/information-on-chemicals/ec-inventory", "fetched_url": URL}
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=False)
        ctx = await browser.new_context(accept_downloads=True)
        page = await ctx.new_page()
        status = "gap: no download after 10 tries"
        for attempt in range(10):
            try:
                async with page.expect_download(timeout=90000) as d:
                    try:
                        await page.goto(URL, timeout=90000)
                    except Exception:
                        pass  # navigation is aborted when a download starts
                dl = await d.value
                out = RAW / "ec_inventory_en.csv"
                await dl.save_as(out)
                status = "ok"
                rec.update(files=[f"raw/{out.name}"], bytes=out.stat().st_size)
                break
            except Exception as exc:
                title = await page.title()
                print(attempt, "no download:", title, str(exc)[:80], flush=True)
                if "Unavailable" not in title and "WAF" not in title and title:
                    status = f"gap: URL answered with a page titled '{title}' and no file"
                    break
                await page.wait_for_timeout(15000)
        await browser.close()
    rec.update(status=status, retrieved=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
    meta["ec_inventory"] = rec
    meta_path.write_text(json.dumps(meta, indent=1))
    print(rec)

if __name__ == "__main__":
    asyncio.run(main())
