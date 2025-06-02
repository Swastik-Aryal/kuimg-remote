import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from autotag.img.crawlers.sites import Sites
import undetected_chromedriver as uc


class CollectLinks:
    def __init__(self, no_gui=False, proxy=None):
        
        chrome_options = uc.ChromeOptions()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        if no_gui:
            chrome_options.add_argument('--headless=new')

        self.browser = uc.Chrome(options=chrome_options, driver_executable_path="/usr/local/bin/chromedriver")
        

        browser_version = "Failed to detect version"
        chromedriver_version = "Failed to detect version"
        major_version_different = False

        if "browserVersion" in self.browser.capabilities:
            browser_version = str(self.browser.capabilities["browserVersion"])

        if "chrome" in self.browser.capabilities:
            if "chromedriverVersion" in self.browser.capabilities["chrome"]:
                chromedriver_version = str(
                    self.browser.capabilities["chrome"]["chromedriverVersion"]
                ).split(" ")[0]

        if browser_version.split(".")[0] != chromedriver_version.split(".")[0]:
            major_version_different = True

        print("_________________________________")
        print("Current web-browser version:\t{}".format(browser_version))
        print("Current chrome-driver version:\t{}".format(chromedriver_version))
        if major_version_different:
            print("warning: Version different")
            print(
                'Download correct version at "http://chromedriver.chromium.org/downloads" and place in "./chromedriver"'
            )
        print("_________________________________")

    def get_scroll(self):
        pos = self.browser.execute_script("return window.pageYOffset;")
        return pos

    def scroller(self, elem):
        last_scroll = 0
        scroll_patience = 0
        NUM_MAX_SCROLL_PATIENCE = 50

        while True:
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.2)

            scroll = self.get_scroll()
            if scroll == last_scroll:
                scroll_patience += 1
            else:
                scroll_patience = 0
                last_scroll = scroll

            if scroll_patience >= NUM_MAX_SCROLL_PATIENCE:
                break

    @staticmethod
    def remove_duplicates(_list):
        return list(dict.fromkeys(_list))

    def search(self, keyword, site_code):
        self.browser.get(Sites.get_domain(site_code).format(keyword))

        time.sleep(1)

        print("Scrolling down")

        elem = self.browser.find_element(By.TAG_NAME, "body")

        self.scroller(elem)

        print("Scraping links")

        imgs = self.browser.find_elements(By.XPATH, Sites.get_XPATH(site_code))

        links = []
        for idx, img in enumerate(imgs):
            try:
                src = img.get_attribute("src")
                links.append(src)

            except Exception as e:
                print(
                    "[Exception occurred while collecting links from {}] {}".format(
                        Sites.get_text(site_code), e
                    )
                )

        links = self.remove_duplicates(links)

        print(
            "Collect links done. Site: {}, Keyword: {}, Total: {}".format(
                Sites.get_text(site_code), keyword, len(links)
            )
        )
        self.browser.close()

        return links
