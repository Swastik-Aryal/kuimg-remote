import os
import shutil
import time
from PIL import Image
from icrawler.builtin import GoogleImageCrawler

def fetch_img(info, f=None) : 
    """
        f is a transformation function that takes a Pillow Image object as input and output a Pillow Image object
    """
    keywords = info.get('keywords', [])
 
    for elem in keywords:
        keyword = elem['key']
        img_count = elem['count']
        target_loc = elem['loc']
        extension = elem['ext']
        file_counter = 0

        google_crawler = GoogleImageCrawler(
            parser_threads=2,
            downloader_threads=4,
            storage={'root_dir': target_loc}
        )

        filters = dict(size=info.get('size', 'large'))

        google_crawler.crawl(
            keyword=keyword, filters=filters, max_num=img_count, file_idx_offset=0)

        # apply transform function to each image generated (if provided)
        if not f : continue

        for root, dirs, files in os.walk(target_loc) : 
            for img_path in (os.path.join(root, _) for _ in files) : 
                img = Image.open(img_path) 
                os.remove(img_path)
                img = f(img)
                img_path = f'{img_path[:img_path.rfind("/")]}/{file_counter:05}.{extension}'
                img.save(img_path)
                file_counter += 1;


if __name__ == '__main__' : 
    target_loc = os.path.join(os.getcwd(), ,'data', 'images')

    # counter chair 
    keywords = [
        { 'key' : 'animals'    , 'count' : 100  , 'loc' : os.path.join(target_loc, 'animals')    , 'ext' : 'jpg'},
        { 'key' : 'table'      , 'count' : 30   , 'loc' : os.path.join(target_loc, 'table')      , 'ext' : 'jpg'}, 
        { 'key' : 'television' , 'count' : 20   , 'loc' : os.path.join(target_loc, 'television') , 'ext' : 'jpg'}, 
        { 'key' : 'frame'      , 'count' : 30   , 'loc' : os.path.join(target_loc, 'frame')      , 'ext' : 'jpg'}, 
        { 'key' : 'pillow'     , 'count' : 10   , 'loc' : os.path.join(target_loc, 'pillow')     , 'ext' : 'jpg'}, 
        { 'key' : 'cushion'    , 'count' : 10   , 'loc' : os.path.join(target_loc, 'cushion')    , 'ext' : 'jpg'}, 
        { 'key' : 'birds'      , 'count' : 100  , 'loc' : os.path.join(target_loc, 'birds')      , 'ext' : 'jpg'}, 
        { 'key' : 'car'        , 'count' : 40   , 'loc' : os.path.join(target_loc, 'car')        , 'ext' : 'jpg'}, 
        { 'key' : 'toaster'    , 'count' : 25   , 'loc' : os.path.join(target_loc, 'toaster')    , 'ext' : 'jpg'}, 
        { 'key' : 'computer'   , 'count' : 10   , 'loc' : os.path.join(target_loc, 'computer')   , 'ext' : 'jpg'}, 
        { 'key' : 'jar'        , 'count' : 10   , 'loc' : os.path.join(target_loc, 'computer')   , 'ext' : 'jpg'}, 
    ]

    # chair 
    keywords = [
        { 'key' :  'arm chair'     , 'count' : 100  , 'loc' : os.path.join(target_loc, 'arm chair'    ), 'ext' : 'jpg'},
        { 'key' :  'bean bag'      , 'count' : 100  , 'loc' : os.path.join(target_loc, 'bean bag'     ), 'ext' : 'jpg'}, 
        { 'key' :  'bench'         , 'count' : 100  , 'loc' : os.path.join(target_loc, 'bench'        ), 'ext' : 'jpg'}, 
        { 'key' :  'chair'         , 'count' : 100  , 'loc' : os.path.join(target_loc, 'chair'        ), 'ext' : 'jpg'}, 
        { 'key' :  'couch'         , 'count' : 100  , 'loc' : os.path.join(target_loc, 'couch'        ), 'ext' : 'jpg'}, 
        { 'key' :  'sofa'          , 'count' : 100  , 'loc' : os.path.join(target_loc, 'sofa'         ), 'ext' : 'jpg'}, 
        { 'key' :  'recliner'      , 'count' : 100  , 'loc' : os.path.join(target_loc, 'recliner'     ), 'ext' : 'jpg'}, 
        { 'key' :  'rocking chair' , 'count' : 100  , 'loc' : os.path.join(target_loc, 'rocking chair'), 'ext' : 'jpg'}, 
        { 'key' :  'gaming chair'  , 'count' : 100  , 'loc' : os.path.join(target_loc, 'gaming chair' ), 'ext' : 'jpg'}, 
        { 'key' :  'wooden chair'  , 'count' : 100  , 'loc' : os.path.join(target_loc, 'wooden chair' ), 'ext' : 'jpg'}, 
    ]

    info = {
        'keywords' : keywords, 
        'size' : 'icon',
    }

    f = lambda img : img.convert('RGB').resize((64,64), resample=Image.Resampling.BILINEAR)
    fetch_img(info, f)
