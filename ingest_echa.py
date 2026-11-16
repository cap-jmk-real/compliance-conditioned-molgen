"""Fetch the ECHA Candidate List, Annex XIV and Annex XVII tables.

The ECHA site sits behind an Azure WAF with a JavaScript challenge and
returns 403 to plain HTTP clients and 503 ("website is busy") to headless
browsers. This script drives a real (headed) Chrome through Playwright,
loads each public list page in pages of 100 rows, and
caches the rendered HTML in raw/ with the source URL and retrieval time.

Usage: DISPLAY=:0 python ingest_echa.py [list_name ...]
"""
import asyncio, datetime, json, pathlib, re, sys
from playwright.async_api import async_playwright

RAW = pathlib.Path(__file__).resolve().parent / "raw"
P = "_disslists_WAR_disslistsportlet_"
LISTS = {
    "candidate_list": "https://echa.europa.eu/candidate-list-table",
    "annex_xiv_authorisation_list": "https://echa.europa.eu/authorisation-list",
    "annex_xvii_restriction_list": "https://echa.europa.eu/substances-restricted-under-reach",
}
CHROME = "/usr/bin/google-chrome"
TRIES = 30


DELTA = 100


def page_url(base, cur):
    return (f"{base}?p_p_id=disslists_WAR_disslistsportlet&p_p_lifecycle=0&p_p_state=normal"
            f"&p_p_mode=view&{P}delta={DELTA}&{P}deltaParamValue={DELTA}&{P}cur={cur}&{P}resetCur=false")


async def load(page, url, label):
    """Return rendered HTML of url, or None after repeated 503/WAF pages."""
    for attempt in range(TRIES):
        try:
            await page.goto(url, wait_until="networkidle", timeout=120000)
        except Exception as exc:  # network error, retry
            print(label, attempt, exc, file=sys.stderr)
            continue
        await page.wait_for_timeout(3000)
        title = await page.title()
        html = await page.content()
        ok = "Unavailable" not in title and "WAF" not in title and "results." in html
        print(label, attempt, title, "ok" if ok else "retry", flush=True)
        if ok:
            return html
        await page.wait_for_timeout(15000)
    return None


async def main():
    RAW.mkdir(exist_ok=True)
    meta_path = RAW / "fetch_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROME, headless=False)
        page = await (await browser.new_context()).new_page()
        wanted = sys.argv[1:] or list(LISTS)
        for name, base in LISTS.items():
            if name not in wanted:
                continue
            files, total, cur, status = [], None, 1, "ok"
            url = page_url(base, 1)
            while True:
                html = await load(page, url, f"{name} p{cur}")
                if html is None:
                    status = f"gap: page {cur} did not load after {TRIES} tries (503 busy / WAF)"
                    break
                f = RAW / f"{name}_p{cur}.html"
                f.write_text(html)
                files.append(f"raw/{f.name}")
                m = re.search(r"of ([0-9]+) results|Showing ([0-9]+) results", html)
                total = int(m.group(1) or m.group(2))
                shown = re.search(r"Showing ([0-9]+) - ([0-9]+) of", html)
                print(name, "page", cur, "shows", shown.groups() if shown else total, flush=True)
                if cur * DELTA >= total:
                    break
                cur += 1
                # The portlet only honours the page number together with the
                # sort and filter parameters it writes into its own pager links.
                nxt = re.search(r'href="([^"]*_cur=%d)"' % cur, html)
                if not nxt:
                    status = f"gap: no pager link to page {cur}"
                    break
                url = nxt.group(1).replace("&amp;", "&")
            meta[name] = {"source_page": base, "status": status, "rows_reported_by_page": total,
                          "retrieved": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                          "files": files}
            meta_path.write_text(json.dumps(meta, indent=1))
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
