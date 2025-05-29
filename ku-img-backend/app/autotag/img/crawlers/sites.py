class Sites:
    GOOGLE = 1
    NAVER = 2
    NATE = 3

    @staticmethod
    def get_text(code):
        if code == Sites.GOOGLE:
            return 'google'
        elif code == Sites.NAVER:
            return 'naver'
        
        ## Domain that obtains watermarked images (Not implemented in crawler right now.) 
        elif code == Sites.NATE:
            return 'nate'
        
    @staticmethod
    def get_domain(code):
        if code == Sites.GOOGLE:
            return "https://www.google.com/search?q={}&source=lnms&tbm=isch"
        elif code == Sites.NAVER:
            return "https://search.naver.com/search.naver?where=image&sm=tab_jum&query={}"
        
        ## Domain that obtains watermarked images (Not implemented in crawler right now.) 
        elif code == Sites.NATE:
            return 'https://search.daum.net/nate?w=img&DA=NTB&enc=utf8&q={}'
       
       
       
    @staticmethod
    def get_XPATH(code):
        if code == Sites.GOOGLE:
            return '//div[@jsname="dTDiAc"]/div[@jsname="qQjpJ"]//img'
        elif code == Sites.NAVER:
            return '//div[@class="tile_item _fe_image_tab_content_tile"]//img[@class="_fe_image_tab_content_thumbnail_image"]'
        
        ## Domain that obtains watermarked images (Not implemented in crawler right now.) 
        elif code == Sites.NATE:
            return '//div[@class="cont_image "]//img'
        