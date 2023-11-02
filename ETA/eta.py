import os

from navertrans import navertrans
import cv2

from ETA.misc import utils


class elite_translate_agents:

    def __init__(self, path, lang = 'en', is_gpu = False, 
                 save_masked = False, page_idx = None):

        self.save_masked = save_masked
        self.is_gpu      = is_gpu
        self.reader      = utils.get_reader(lang, is_gpu)
        self.lang        = lang
        paper            = utils.read_paper(path)

        if isinstance(page_idx, list):
            start_idx, end_idx = page_idx
            self.paper         = paper[start_idx - 1 : end_idx]

        elif isinstance(page_idx, int):
            self.paper = [paper[page_idx - 1]]


    def translate(self):

        translated = []
        for idx, page in enumerate(self.paper, 1):

            page = utils.get_masked_page(page)

            if self.save_masked:
                os.makedirs('masked_pages', exist_ok = True)
                cv2.imwrite(f'masked_pages/page_{str(idx).zfill(2)}', page)
            
            text = utils.read_text(page, self.reader)
            text = navertrans.translate(text, src_lan = 'en', tar_lan = 'ko')
            translated.append(text)

        return '\n'.join(translated)



    


