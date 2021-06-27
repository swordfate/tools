import os

import pdfplumber
from pdfminer.pdfparser import PDFSyntaxError

# 输入文件夹路径，输出该文件夹下一共有多少pdf和一共的页数

pdfsdir = input('输入文件夹路径：')


pdflst = []
total_pages = 0

def get_pdf_page(pdf_path):
    try:
        f = pdfplumber.open(pdf_path)
        page = len(f.pages)
    except PDFSyntaxError:
        page = 0
    return page

def countPdfs(root_path):
    global pdflst, total_pages

    fileList = os.listdir(root_path)
    for file in fileList:
        if os.path.isdir(os.path.join(root_path,file)):
            countPdfs(os.path.join(root_path, file))
        else:
            if os.path.splitext(file)[-1] == '.pdf':
                file_path = os.path.join(root_path,file)
                total_pages += get_pdf_page(file_path)
                pdflst.append(file_path)

countPdfs(pdfsdir)

print('Total n_pdf:', len(pdflst))
print('Total pages:', total_pages)