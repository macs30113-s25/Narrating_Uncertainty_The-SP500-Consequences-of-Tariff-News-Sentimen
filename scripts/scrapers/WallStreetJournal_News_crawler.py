from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from urllib.parse import urljoin
from newspaper import Article
import urllib.parse
import time
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

from selenium.webdriver.support.ui import WebDriverWait


keywords = ["import tax", "tariff", "trade war", "customs duty", "import duty", "export tax", "trade policy", "international trade", "global trade", "trade agreement"]
cookie_string = """_pubcid=11019a7e-f32a-449d-8064-504ade5e1b39; _pubcid_cst=DCwOLBEsaQ%3D%3D; _lr_retry_request=true; _lr_env_src_ats=false; _sp_su=false; ca_r=www.wsj.com; AMCVS_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1; AMCV_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1585540135%7CMCIDTS%7C20239%7CMCMID%7C71465904542642135216855030565737395970%7CMCAAMLH-1749204350%7C11%7CMCAAMB-1749204350%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1748606751s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.4.0; _ncg_domain_id_=0ecc1a08-22d3-4aa5-89ee-5920d74f0c9f.1.1748599550.1780135550; s_cc=true; _pcid=%7B%22browserId%22%3A%22mban0wizhmhxy2ya%22%7D; _dj_ses.9183=*; _dj_sp_id=041a17d0-c77c-442a-8df1-c1729ec9a280; _ga=GA1.1.414346548.1748599554; _scor_uid=b53799a7a7ef4d1db34130bf999df852; ajs_anonymous_id=1900c43a-a690-4475-a20d-0bb22edfd1a2; _fbp=fb.1.1748599557475.684703879; _meta_facebookTag_sync=1748599557475; _meta_googleAdsSegments_library_loaded=1748599557478; dicbo_id=%7B%22dicbo_fetch%22%3A1748599557659%7D; __pat=-14400000; xbc=%7Bkpcd%7DChBtYmFuMHdpemhtaHh5MnlhEgpLS2JncXBCbHB1GjxrYnl2YWUzRzZKQmt5NUVsSmJrOXZONHNEbWwwR1pZSUR4dGFjWkNGdklQOTNIdU1xY1dSVjI2SFdQYWUgAA; cX_P=mban0wizhmhxy2ya; _ncg_g_id_=6e0da41b-44d5-4080-b976-b7de56503e4d.3.1748511287.1780135550; _meta_cross_domain_id=45d19532-d0e3-4d4a-84d7-112b12ca6cc2; _meta_cross_domain_recheck=1748599558316; _fbp=fb.1.1748599557475.684703879; LANG=en_US; LANG_CHANGED=en_US; _gcl_au=1.1.1143153906.1748599559; _pin_unauth=dWlkPVpEbGpOMlU1TnpZdE5UaG1ZaTAwWVdJMkxUZ3hOR1F0TnpBMk5UQXpOREF6TkdZeg; djcs_route=c1abfd6a-3172-4630-9d05-75db5c2be505; ca_rt=aWl8rN532AAus5_OQpzAVA.fUy7-De0twh2iEr_BVv0Pr3AkBXWmhyeVUTHK115-o9_sZznvqhX8qNtSk-85MX5vLUxigkkfDJWpS7RqP08eE5Bo8bzDXy9yKVnL7_X0W4; ca_id=0f7b9586-b922-4403-8396-039962f63385.eJw1kNtSwjAQht8l1xSaJmkTrgRbsVIOAiozjtNJk7REykGaUhzHdzccvNv9_t1_Dz-g0Ee1Tbd8o0AXLPW2UKAFcr7R5fc_TbRFasN1aZPydK65q8VKC17s2krWVq1rLa0IGXUJzbCjaIYcjKHncCiJgwXMqC-hL3Jkq82Bi_W1wfOhn-cZ4oIyTBWnMkPUZyTLXEVchBBUOXcZkgFUWHg5oh7LBSYcChYgbM0Ou1JVoPsOHmZRNIsGTjwO49c4fOklVn2bPznxtBfewt44nE3iMF30-km0uMH7xfIWjSb9OImuCfhoAV6bVWr0-QswwJRAiIjXAtVld5SZIlmbiA5TssPhdIIHfTIaPcMxluHQnfle5K0d94vT9fHxa-7Iaj_fF9ZcHBQ3SqbcnG0RJTiALmoBfQV2DmMksECd9hfAXAQvQFf2UrAyZl91O52madpN9dkWu01HlFptDfj9Awomhcs.HUSRwUBWyt4K24INnNUUg2kGDuvtg9pzKqmB3Bjlp3wXhwk94ZkDInNiW2UGwg-cKjn-O6wCvIdO4ghQ__SL9ojCLYtzBEOEABlwq88oA0oZCUfSDL5GoV3pVU6ns-_elpfVfCwpoOF1J-kjV9-5OZ2ea3xpssbuAKAzWdtYXg5zQJjoC4bv1RXPBC7WAcf2XRpSwi01ZjQbR9qbfaDoTaSfph50vGpRe6-KKM6CwCwoK78NkMaaik-yj8GBYGaZN7V0CAMnFGG2FoZeyV_MTt9ZeoebKOfEiopgMAXm41ks2XuA9a0UtbAH8iIdaapPbzsQlx69eoMqbZJMUPyyiA; TR=V2-12616ffb3ac8948ea8db38695bb0e503331efa093d71e4c2f3829fc45a1c9734; usr_prof_v2=eyJhIjp7ImVkIjoidHJ1ZSIsInIiOiIyMDI1LTAyLTAyVDAwOjAwOjAwLjAwMFoiLCJlIjoiMjA0MC0wMi0wMVQwMDowMDowMC4wMDBaIiwicyI6IjIwMjUtMDItMDJUMDA6MDA6MDAuMDAwWiIsInNiIjoiRWR1Y2F0aW9uYWwiLCJmIjoiMiBZZWFyIiwibyI6IlN0dWRlbnQgRGlnaXRhbCBQYWNrIiwiYXQiOiJTVFVERU5UIiwidCI6M30sImNwIjp7ImVjIjoiR3Jvd2luZyIsInBjIjowLjAwMjEyLCJwc3IiOjAuMTIyMzYsInRkIjoxMTYsImFkIjozLCJxYyI6MywicW8iOjIsInNjZW4iOnsiY2hlIjowLjAwMzgxLCJjaG4iOjAuMDAxODksImNoYSI6MC4wMDUyNSwiY2hwIjowLjAwMjEyfX0sIm9wIjp7ImkiOiI0NTU4MDA4MCIsImoiOnsiamYiOiJkYSJ9fSwiaWMiOjV9; ab_uuid=d062f993-1fcd-4422-ab11-1a0586750ece; DJSESSION=country%3Djp%7C%7Ccontinent%3Das%7C%7Cregion%3D; wsjregion=na%2Cus; gdprApplies=false; ccpaApplies=false; vcdpaApplies=false; regulationApplies=gdpr%3Afalse%2Ccpra%3Afalse%2Cvcdpa%3Afalse; __tac=eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczovL3d3dy5waWFuby5pbyIsImF1ZCI6IktLYmdxcEJscHUiLCJzdWIiOiJ7XCJ1XCI6XCIxMjYxNmZmYjNhYzg5NDhlYThkYjM4Njk1YmIwZTUwMzMzMWVmYTA5M2Q3MWU0YzJmMzgyOWZjNDVhMWM5NzM0XCJ9IiwiZXhwIjoxNzUxMTkxNTgwLCJpYXQiOjE3NDg1OTk1ODB9.Uw5hMd5Hd1KWQz3ITAG5DWzsNvk5vQjjujd9uDzdizA; __tae=1748599580139; _pctx=%7Bu%7DN4IgrgzgpgThIC5QEYBMA2Z6Bm2BGAzAIYDGAHAJwAsZURZAJoWehQKx54AMUbXBA5FGxEuFAgwDsQqiVTYCZVBWwkqbIshIVJBKolAAHGMICWAD0QgA7hABWIADQgALgE9DUKwDUAGiABfAOdIWABlFyIXSCsTAHNTCBdYKAYnEAhTZIBJNIQQClQigmQWZEU2VBpWLi5AoA; _pcus=eyJ1c2VyU2VnbWVudHMiOnsiQ09NUE9TRVIxWCI6eyJzZWdtZW50cyI6WyJhOXlqZWVuYWQ3eXEiLCJhOXlqZWVuYWQ4MWkiLCJhYTltaHh6dmkwYXgiLCJhYTlvc2tnb2pvYnkiLCJhYTlxNmU0czFuazgiLCJhYTlxNmU0czFuazkiLCJhYTlxNmU0czFua2IiLCJhYWt2Ym1waGZvZ2IiLCJhYXZzcDBmcmxpdWYiLCJhYXZzcDBmcmxqYTEiLCJhYXZzcDBmcmxqZTUiLCJhYXZzcDBmcmxqbmsiLCJhYXZ3ZGdycW02d3EiLCJhYXZ3ZGdycW02d3MiLCJhYXcwbGJyb3dzdWwiXX19fQ%3D%3D; optimizelySession=1748599586037; search_test_bucket_v2=olympia_semantic_search; utag_main=v_id:019720a7b104005af10a0fca12f80206f007306700a83$_sn:1$_se:5$_ss:0$_st:1748601416027$ses_id:1748599550213%3Bexp-session$_pn:3%3Bexp-session$_prevpage:WSJ_ResearchTools_Search%3Bexp-1748603216030$vapi_domain:wsj.com; _uetsid=b60099e03b6c11f0a95aaf063b24e611; _uetvid=b600b2a03b6c11f0a589b99bf935e74c; _ga_K2H7B9JRSS=GS2.1.s1748599554$o1$g1$t1748599616$j60$l0$h0; _rdt_uuid=1748599556095.c37dc721-e420-429b-b723-d3fa466812f1; datadome=t2pdQFnbfCDWulqrawd8QmCvG2Ry7RX98k2ELGMHPKG0CB7sEkHyytfcY344rZU7cfBfukhBATpEVGwVUjgNBvYDYfeTUoDtt6ebgDETwn8MOwY8t9etyEy0KRQJDNnJ; __tbc=%7Bkpcd%7DChBtYmFuMHdpemhtaHh5MnlhEgpLS2JncXBCbHB1GjxrYnl2YWUzRzZKQmt5NUVsSmJrOXZONHNEbWwwR1pZSUR4dGFjWkNGdklQOTNIdU1xY1dSVjI2SFdQYWUgAA; s_ppv=%5B%5BB%5D%5D; s_tp=6261"""

options = Options()
driver = webdriver.Chrome(options=options)
from dateutil import parser

def extract_date_from_meta(driver):
    try:
        meta = driver.find_element(By.XPATH, '//meta[@name="article.published"]')
        content = meta.get_attribute("content")
        if content:
            dt = parser.parse(content)
            return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


driver.get("https://www.wsj.com")
driver.delete_all_cookies()
for part in cookie_string.strip().split("; "):
    if "=" in part:
        name, value = part.split("=", 1)
        cookie = {
            "name": name,
            "value": urllib.parse.unquote(value),
            "domain": ".wsj.com",
            "path": "/",
        }
        try:
            driver.add_cookie(cookie)
        except Exception as e:
            print(f"Failed to add cookie: {name} -> {e}")


driver.get("https://www.wsj.com")

all_links = set()

START_PAGE = 1
END_PAGE = 1

for keyword in keywords:
    for page in range(START_PAGE, END_PAGE + 1):
        url = (
            f"https://www.wsj.com/search?query={urllib.parse.quote(keyword)}"
            f"&mod=searchresults_viewallresults"
            f"&dateRange=custom&dateFrom=2020-01-01&dateTo=2020-03-31"  # Adjust date range as needed
            + (f"&page={page}" if page > 1 else "")  # Adjust the number of pages for scraping
        )
        driver.get(url)
        time.sleep(2)

        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

        print(f"Current keyword: {keyword}, page {page}")

        wait = WebDriverWait(driver, 10)
        for i in range(1, 21):
            base_xpath = f'//*[@id="__next"]/div/main/div[2]/div[1]/div[2]/div[{i}]'

            try:
                href = driver.find_element(By.XPATH,
                                           base_xpath + '/div/div[2]/div/div/h3/a'
                                           ).get_attribute('href')
                publish_date= None
            except NoSuchElementException:
                continue

            all_links.add((href, publish_date))
            print(href)

# ----------------- Extract article content -----------------
results = []
for idx, (link, publish_date) in enumerate(all_links, start=1):
    try:
        driver.get(link)
        time.sleep(3)
        if not publish_date:
            publish_date = extract_date_from_meta(driver)
            if publish_date:
                print(f"[{idx}] Supplemented date from meta: {publish_date}")
            else:
                print(f"[{idx}] Unable to get publish date, skipping: {link}")
                continue

        html = driver.page_source
        article = Article(link)
        article.download_state = 2
        article.set_html(html)
        article.parse()

        title = article.title.strip()
        content = article.text.strip()
        summary = article.meta_description or ""

        results.append({
            "source": "WSJ",
            "title": title,
            "summary": summary,
            "link": link,
            "content": content,
            "published": publish_date
        })

        print(f"[{idx}] Successfully scraped: {title}")

    except Exception as e:
        print(f"[{idx}] Fail: {link} -> {e}")


# ----------------- Save as JSON grouped by date -----------------
import os
import json
from collections import defaultdict

articles_by_date = defaultdict(list)
for article in results:
    pub_date = article["published"]
    articles_by_date[pub_date].append(article)

os.makedirs("articles_by_date11112", exist_ok=True)

for pub_date, articles in articles_by_date.items():
    if pub_date:
        filename = f"articles_by_date11112/{pub_date.replace('-', '')}.json"
    else:
        filename = "articles_by_date11112/unknown_date.json"

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []

    # Merge
    combined = existing + articles

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

print("Save completed.")
driver.quit()
