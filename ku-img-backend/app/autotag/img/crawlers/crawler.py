import os
import requests
import shutil
from multiprocessing import Pool
from autotag.img.crawlers.collect_links import CollectLinks
import base64
import random
from autotag.img.crawlers.duplicate_checker import Duplicates
from autotag.img.crawlers.sites import Sites


class Crawler:
    def __init__(
        self,
        keyword,
        skip_already_exist=True,
        download_path="download",
        no_gui=False,
        limit=0,
        proxy_list=None,
    ):
        self.keyword = keyword
        self.skip = skip_already_exist
        self.n_threads = 2
        self.download_path = download_path
        self.no_gui = no_gui
        self.limit = limit
        self.proxy_list = proxy_list if proxy_list and len(proxy_list) > 0 else None
        self.dupe_check_done = False

    @staticmethod
    def get_extension_from_link(link, default="jpg"):
        splits = str(link).split(".")
        if len(splits) == 0:
            return default
        ext = splits[-1].lower()
        if ext == "jpg" or ext == "jpeg":
            return "jpg"
        elif ext == "gif":
            return "gif"
        elif ext == "png":
            return "png"
        else:
            return default

    @staticmethod
    def validate_image(path):
        _, ext = os.path.splitext(path)
        ext = ext.lower().lstrip(".")

        if not ext:
            return None

        if ext == "jpeg":
            ext = "jpg"
        return ext

    @staticmethod
    def make_dir(current_path, skip_value):
        if os.path.exists(current_path) and not skip_value:
            shutil.rmtree(current_path)
            os.makedirs(current_path)
        elif not os.path.exists(current_path):
            os.makedirs(current_path)

    @staticmethod
    def save_object_to_file(object, file_path, is_base64=False):
        try:
            with open("{}".format(file_path), "wb") as file:
                if is_base64:
                    file.write(object)
                else:
                    shutil.copyfileobj(object.raw, file)
        except Exception as e:
            print("Save failed - {}".format(e))

    @staticmethod
    def base64_to_object(src):
        header, encoded = str(src).split(",", 1)
        data = base64.decodebytes(bytes(encoded, encoding="utf-8"))
        return data

    @staticmethod
    def delete_extras(directory, num):
        files = os.listdir(directory)
        file_count = len(files)
        if file_count > num:
            files_to_delete = files[num:]
            for file in files_to_delete:
                os.remove(os.path.join(directory, file))

    def download_images(self, keyword, links, site_name):
        success_count = 0
        max_count = self.limit if self.limit > 0 else 10000

        for index, link in enumerate(links):
            self.dupe_check_done = False

            try:
                print(
                    "Downloading {} from {}: {}".format(
                        keyword, site_name, success_count + 1
                    )
                )

                if str(link).startswith("data:image/jpeg;base64"):
                    response = self.base64_to_object(link)
                    ext = "jpg"
                    is_base64 = True
                elif str(link).startswith("data:image/png;base64"):
                    response = self.base64_to_object(link)
                    ext = "png"
                    is_base64 = True
                else:
                    response = requests.get(link, stream=True, timeout=10)
                    ext = self.get_extension_from_link(link)
                    is_base64 = False

                no_ext_path = "{}/{}_{}".format(
                    self.download_path, site_name, str(index).zfill(4)
                )
                path = no_ext_path + "." + ext
                self.save_object_to_file(response, path, is_base64=is_base64)

                success_count += 1
                del response

                ext2 = self.validate_image(path)
                if ext2 is None:
                    print("Unreadable file - {}".format(link))
                    os.remove(path)
                    success_count -= 1
                else:
                    if ext != ext2:
                        path2 = no_ext_path + "." + ext2
                        os.rename(path, path2)
                        print("Renamed extension {} -> {}".format(ext, ext2))

            except KeyboardInterrupt:
                break

            except Exception as e:
                print("Download failed - ", e)
                continue

            if len(os.listdir(self.download_path)) >= max_count:
                try:
                    Duplicates.remove_duplicates(self.download_path)
                except:
                    pass
                if len(os.listdir(self.download_path)) >= max_count:
                    self.dupe_check_done = True
                    break

    def download_from_site(self, keyword, site_code):
        site_name = Sites.get_text(site_code)

        try:
            proxy = None
            if self.proxy_list:
                proxy = random.choice(self.proxy_list)
            collect = CollectLinks(
                no_gui=self.no_gui, proxy=proxy
            )  # initialize chrome driver
        except Exception as e:
            print("Error occurred while initializing chromedriver - {}".format(e))
            return

        try:
            print("Collecting links... {} from {}".format(keyword, site_name))
            links = collect.search(keyword, site_code)

            print(
                "Downloading images from collected links... {} from {}".format(
                    keyword, site_name
                )
            )
            self.download_images(keyword, links, site_name)

        except Exception as e:
            print("Exception :{} - {}".format(keyword, e))
            return

    def download(self, args):
        self.download_from_site(keyword=args[0], site_code=args[1])

    def crawl(self):
        tasks = []
        done = os.path.exists(self.download_path)
        if done and self.skip:
            print(
                "Skipping image download since folder: '{}' exists.".format(
                    self.download_path
                )
            )
            return tasks
        else:
            self.make_dir(self.download_path, self.skip)
            tasks.append([self.keyword, Sites.GOOGLE])
            tasks.append([self.keyword, Sites.NAVER])
            # tasks.append([self.keyword, Sites.NATE])

        try:
            pool = Pool(self.n_threads)
            pool.map(self.download, tasks)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        else:
            pool.terminate()
            pool.join()

        print("done till here.")

        if not self.dupe_check_done:
            Duplicates.remove_duplicates(self.download_path)

        if self.limit > 0:
            self.delete_extras(self.download_path, self.limit)

        print("No more duplicates found.")
        print(
            "Task ended. Downloaded {} images at location {}".format(
                len(os.listdir(self.download_path)), self.download_path
            )
        )


if __name__ == "__main__":
    # parser.add_argument('--proxy-list', type=str, default='',
    #                     help='The comma separated proxy list like: "socks://127.0.0.1:1080,http://127.0.0.1:1081". '
    #                          'Every thread will randomly choose one from the list.')

    word = "dog"

    crawler = Crawler(
        keyword=word, skip_already_exist=False, no_gui=True, limit=10, proxy_list=None
    )
    crawler.crawl()
