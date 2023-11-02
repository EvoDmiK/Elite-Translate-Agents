from pdf2image import convert_from_path
from easyocr import Reader
import numpy as np
import cv2

## easyOCR reader 불러오는 함수
## gpu를 사용하지 않으면 엄청 느리다..
def get_reader(lang, is_gpu = False):

    lang = [lang] if isinstance(lang, str) else lang
    return Reader(lang_list = lang, gpu = is_gpu)


## 페이지 내 figure, table을 가려주는 함수
## figure, table에 글자가 있는 경우 OCR이 읽어 문장이 이상해지는 경우가 있어 정의함.
def get_masked_page(page):

    page     = np.array(page)
    gray     = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    _, bin_  = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    H, W     = page.shape[:2]

    ## 페이지 내 글씨 이외의 figure나 table의 contour를 찾아주는 부분.
    conts, _ = cv2.findContours(bin_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ## 페이지 내에 있는 contour의 넓이를 계산해서 리스트에 저장하는 부분.
    areas    = [cv2.contourArea(cont) for cont in conts]
    conts    = [(area, cont) for area, cont in zip(areas, conts) 
                if 1_000 < area < H * W - 5_000]

    if conts:
        ## 면적이 넓은 순으로 정렬
        conts = sorted(conts, key = lambda x: x[0], reverse = True)
        for _, cont in conts:

            x, y, w, h = cv2.boundingRect(cont)
            
            ## figure, table에 검정색으로 마스킹
            ## 양 옆, 위 아래로 조금 여유롭게 마스킹하도록 설정.
            cv2.rectangle(gray, (x - 10, y), ((x + w) + 10, y + h), (0, 0, 0), -1)

    return gray


## easyOCR을 이용해 페이지 내 텍스트를 읽어오는 함수
def read_text(page, reader):
    
    texts = reader.readtext(page)

    ## OCR로 읽어들인 글자 중 글자 수가 1글자 이상인 텍스트만 반환
    return ' '.join([text[1] for text in texts if len(text[1]) != 1])


read_paper  = lambda path: convert_from_path(path, fmt = 'jpg')


