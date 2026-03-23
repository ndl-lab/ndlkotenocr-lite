import logging
#logging.basicConfig(filename='debug.log', encoding='utf-8',level=logging.DEBUG)
import flet as ft
from typing import List, Dict, Tuple
import sys
import os
import numpy as np
from PIL import Image, ImageGrab
from pathlib import Path
sys.path.append(os.path.join(".","src"))
import ocr
from tools.ndlkoten2tei import convert_tei
import xml.etree.ElementTree as ET
import time
from concurrent.futures import ThreadPoolExecutor
import time
import json
import shutil
import argparse
import yaml
import io
import glob
import pypdfium2
import base64
import ctypes
from io import BytesIO

from reading_order.xy_cut.eval import eval_xml
from ndl_parser import convert_to_xml_string3
from ndl_parser import categories_org_name_index

# --- 翻訳辞書 ---
TRANSLATIONS = {
    "main_explain":{
        "ja":"処理対象（古典籍）と出力先を選択して「OCR」ボタンを押してください",
        "en":"Please select target images(pre-modern materials) and output folder, then press OCR"
    },
    "imageselector_cropocr_btn":{
        "ja":"切り抜きOCR",
        "en":"Crop OCR"
    },
    "imageselector_zoom_btn":{
        "ja":"拡大する",
        "en":"Zoom"
    },
    "imageselector_prev_btn":{
        "ja":"前の画像",
        "en":"Prev Image"
    },
    "imageselector_next_btn":{
        "ja":"次の画像",
        "en":"Next Image"
    },
    "imageselector_result_title":{
        "ja":"切り抜きOCR結果",
        "en":"OCR Result"
    },
    "capturetool_retry_btn":{
        "ja":"再キャプチャ",
        "en":"Retake"
    },
    "capturetool_fixregion":{
        "ja":"範囲固定",
        "en":"Fix Region"
    },
    "capturetool_ocr_button":{
        "ja":"OCR実行",
        "en":"OCR"
    },
    "capturetool_result_title":{
        "ja":"キャプチャ結果",
        "en":"Captured Image"
    },
    "capturetool_resultocr_title":{
        "ja":"OCR結果",
        "en":"OCR Result"
    },
    "main_file_upload_btn":{
        "ja":"画像ファイルを処理する",
        "en":"Process an Image"
    },
    "main_directory_upload_btn":{
        "ja":"フォルダ内の画像を処理する",
        "en":"Process Images in a Folder",
    },
    "main_directory_output_btn":{
        "ja":"出力先を選択する",
        "en":"Select Output Directory",
    },
    "main_prev_btn":{
        "ja":"前の画像",
        "en":"Prev Image"
    },
    "main_next_btn":{
        "ja":"次の画像",
        "en":"Next Image"
    },
    "main_cap_btn":{
        "ja":"キャプチャモード",
        "en":"ScreenCapture Mode"
    },
    "main_customize_btn":{
        "ja":"出力形式の選択",
        "en":"Custom Setting"
    },
    "main_target_label":{
        "ja":"処理対象：",
        "en":"Target Path:"
    },
    "main_output_label":{
        "ja":"出力先：",
        "en":"Output Path:"
    },
    "main_preview_label":{
        "ja":"処理結果プレビュー",
        "en":"Preview of Result"
    },
    "main_visualize_label":{
        "ja":"認識箇所の可視化画像を保存する",
        "en":"Save Visualization Results"
    },
    "customize_dlg_modal_title":{
        "ja":"設定",
        "en":"Setting"        
    },
    "customize_dlg_modal_explain":{
        "ja":"出力形式を選択してください",
        "en":"Please select output format"        
    },
    "common_cancel":{
        "ja":"キャンセル",
        "en":"Dismiss",
    },
    "common_close":{
        "ja":"閉じる",
        "en":"Close",
    }
}

name = "NDLkotenOCR-Lite-GUI"
PDFTMPPATH = "4ab7ecc3-53fb-b3e7-64e8-a809b5a483d2"

def get_windows_scale_factor():
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return dpi / 96.0  # 標準が96dpiなので、それを割って倍率を出す
    except:
        return 1.0

# ocr.pyにprocess_detectorがない場合のヘルパー関数
def process_detector(detector, inputname, npimage, outputpath, issaveimg=False):
    detections = detector.detect(npimage)
    classeslist = list(detector.classes.values())
    if issaveimg:
        drawimage = npimage.copy()
        pil_image = detector.draw_detections(drawimage, detections=detections)
        os.makedirs(outputpath, exist_ok=True)
        output_path = os.path.join(outputpath, f"viz_{Path(inputname).name}")
        # print(f"[INFO] Saving result on {output_path}")
        pil_image.save(output_path)
    return detections, classeslist

class RecogLine:
    def __init__(self,npimg:np.ndarray,idx:float,pred_char_cnt:int,pred_str:str=""):
        self.npimg = npimg
        self.idx   = idx
        self.pred_char_cnt = pred_char_cnt
        self.pred_str = pred_str
    def __lt__(self, other):  
        return self.idx < other.idx
    
def process_cascade(alllineobj:RecogLine,recognizer30,recognizer50,recognizer100,is_cascade=True):
    targetdflist30,targetdflist50,targetdflist100,targetdflist200=[],[],[],[]
    for lineobj in alllineobj:
        if lineobj.pred_char_cnt==3 and is_cascade:
            targetdflist30.append(lineobj)
        elif lineobj.pred_char_cnt==2 and is_cascade:
            targetdflist50.append(lineobj)
        else:
            targetdflist100.append(lineobj)
    targetdflistall=[]
    with ThreadPoolExecutor(thread_name_prefix="thread") as executor:
        resultlines30,resultlines50,resultlines100,resultlines200=[],[],[],[]
        if len(targetdflist30)>0:
            resultlines30 = executor.map(recognizer30.read, [t.npimg for t in targetdflist30])
            resultlines30 = list(resultlines30)
        for i in range(len(targetdflist30)):
            pred_str=resultlines30[i]
            lineobj=targetdflist30[i]
            if len(pred_str)>=25:
                targetdflist50.append(lineobj)
            else:
                lineobj.pred_str=pred_str
                targetdflistall.append(lineobj)
        if len(targetdflist50)>0:
            resultlines50 = executor.map(recognizer50.read, [t.npimg for t in targetdflist50])
            resultlines50 = list(resultlines50)
        for i in range(len(targetdflist50)):
            pred_str=resultlines50[i]
            lineobj=targetdflist50[i]
            if len(pred_str)>=45:
                targetdflist100.append(lineobj)
            else:
                lineobj.pred_str=pred_str
                targetdflistall.append(lineobj)
        if len(targetdflist100)>0:
            resultlines100 = executor.map(recognizer100.read, [t.npimg for t in targetdflist100])
            resultlines100 = list(resultlines100)
        for i in range(len(targetdflist100)):
            pred_str=resultlines100[i]
            lineobj=targetdflist100[i]
            lineobj.pred_str=pred_str
            if len(pred_str)>=98 and lineobj.npimg.shape[0]<lineobj.npimg.shape[1]:
                baseimg=lineobj.npimg
                tmplineobj_1=RecogLine(npimg=baseimg[:,:baseimg.shape[1]//2,:],idx=lineobj.idx,pred_char_cnt=100)
                tmplineobj_2=RecogLine(npimg=baseimg[:,baseimg.shape[1]//2:,:],idx=lineobj.idx,pred_char_cnt=100)
                targetdflist200.append(tmplineobj_1)
                targetdflist200.append(tmplineobj_2)
            else:
                targetdflistall.append(lineobj)
        if len(targetdflist200)>0:
            resultlines200 = executor.map(recognizer100.read, [t.npimg for t in targetdflist200])
            resultlines200 = list(resultlines200)
            for i in range(0,len(targetdflist200)-1,2):
                ia=targetdflist200[i]
                lineobj=RecogLine(npimg=None,idx=ia.idx,pred_char_cnt=100,pred_str=resultlines200[i]+resultlines200[i+1])
                targetdflistall.append(lineobj)
        targetdflistall=sorted(targetdflistall)
        resultlinesall=[t.pred_str for t in targetdflistall]
    return resultlinesall


class ImageSelector:
    def __init__(self, page: ft.Page,config_obj:Dict,detector=None, recognizer30=None,recognizer50=None,recognizer100=None,outputdirpath=None,width: int =600, height: int = 600):
        self.cnt=0
        self.page = page
        self.config_obj=config_obj
        self.langcode=config_obj["langcode"]
        self.inputpathlist=[]
        self.outputdirpath=outputdirpath
        
        self.image_src = "dummy.dat"
        self.dialog_width = width
        self.dialog_height = height
        self.page_index=0
        self.detector=detector
        self.recognizer30=recognizer30
        self.recognizer50=recognizer50
        self.recognizer100=recognizer100

        self.start_x = 0
        self.start_y = 0

        self.selection_box = ft.Container(
            left=0, top=0, width=0, height=0,
            border=ft.border.all(2, ft.Colors.BLUE),
            bgcolor=ft.Colors.TRANSPARENT,
        )

        self.overlay = ft.GestureDetector(
            content=ft.Container(
                width=self.dialog_width,
                height=self.dialog_height,
                bgcolor=ft.Colors.TRANSPARENT,
            ),
            on_pan_start=self.pan_start,
            on_pan_update=self.pan_update,
            on_pan_end=self.pan_end,
        )
        self.img=ft.Image(src=self.image_src, width=self.dialog_width, height=self.dialog_height,fit=ft.ImageFit.CONTAIN)
        self.imgzm=ft.Image(src=self.image_src, width=self.dialog_width, height=self.dialog_height,fit=ft.ImageFit.CONTAIN)
        
        self.image_stack = ft.Stack(
            width=self.dialog_width,
            height=self.dialog_height,
            controls=[
                self.img,
                self.selection_box,
                self.overlay,
            ]
        )
        self.cropocr_btn=ft.ElevatedButton(TRANSLATIONS["imageselector_cropocr_btn"][self.langcode], on_click=self.crop_region)
        self.dialog = ft.AlertDialog(
            modal=True,
            content=self.image_stack,
            actions=[
                ft.ElevatedButton(TRANSLATIONS["imageselector_zoom_btn"][self.langcode],icon=ft.Icons.ZOOM_IN, on_click=self.open_zoom_page),
                ft.ElevatedButton(TRANSLATIONS["imageselector_prev_btn"][self.langcode], on_click=self.prev_page),
                ft.ElevatedButton(TRANSLATIONS["imageselector_next_btn"][self.langcode], on_click=self.next_page),
                self.cropocr_btn,
                ft.ElevatedButton(TRANSLATIONS["common_cancel"][self.langcode], on_click=self.close_dialog)
            ],
        )
        zoom_img=ft.InteractiveViewer(
            min_scale=1,
            max_scale=10,
            boundary_margin=ft.margin.all(20),
            content=self.imgzm)
        
        self.zoom_dialog=ft.AlertDialog(
            modal=True,
            content=zoom_img,
            actions=[
                ft.ElevatedButton(TRANSLATIONS["common_cancel"][self.langcode], on_click=self.close_zoom_page)
            ]
        )
        self.resulttext=ft.Text(value="",selectable=True,color=ft.Colors.BLACK)
        
        self.crop_image=ft.Image(src=self.image_src, width=300, height=300,fit=ft.ImageFit.CONTAIN)
        crop_image_col = ft.Column(
            controls=[self.crop_image],
            width=300,
            height=300,
            expand=False
        )
        self.crop_image_int=ft.InteractiveViewer(
            min_scale=1,
            max_scale=5,
            boundary_margin=ft.margin.all(20),
            content=crop_image_col)
        self.result_text_col = ft.Column(
            controls=[self.resulttext],
            scroll=ft.ScrollMode.ALWAYS,
            width=800,
            height=300,
            expand=False
        )
        
        self.result_dialog= ft.AlertDialog(
            title=ft.Text(TRANSLATIONS["imageselector_result_title"][self.langcode]),
            modal=True,
            content=ft.Row([self.crop_image_int,self.result_text_col]),
            actions=[
                ft.ElevatedButton("OK", on_click=self.close_result_page)
            ]
        )
    def open_result_page(self):
        self.dialog.open = False
        self.result_dialog.open = True
        self.page.overlay.append(self.result_dialog)
        self.page.update()

    def close_result_page(self,e):
        self.result_dialog.open = False
        self.dialog.open = True
        self.page.update()

    def set_image(self, inputpathlist):
        self.cnt=0
        self.inputpathlist=inputpathlist
        self.image_src=inputpathlist[self.page_index]
        self.img.src = inputpathlist[self.page_index]
        self.imgzm.src = inputpathlist[self.page_index]
        self.page.update()

    def set_outputdir(self,outputdirpath):
        self.outputdirpath=outputdirpath


    def open_zoom_page(self,e):
        self.dialog.open = False
        if not self.zoom_dialog in self.page.overlay:
            self.page.overlay.append(self.zoom_dialog)
        self.zoom_dialog.open = True
        self.page.update()

    def close_zoom_page(self, e):
        self.zoom_dialog.open = False
        self.dialog.open = True
        self.page.update()

    def pan_start(self, e: ft.DragStartEvent):
        self.start_x = e.local_x
        self.start_y = e.local_y
        self.selection_box.left = self.start_x
        self.selection_box.top = self.start_y
        self.selection_box.width = 0
        self.selection_box.height = 0
        self.page.update()

    def pan_update(self, e: ft.DragUpdateEvent):
        cur_x, cur_y = e.local_x, e.local_y
        left = min(self.start_x, cur_x)
        top = min(self.start_y, cur_y)
        width = abs(cur_x - self.start_x)
        height = abs(cur_y - self.start_y)
        self.selection_box.left = left
        self.selection_box.top = top
        self.selection_box.width = width
        self.selection_box.height = height
        self.page.update()

    def pan_end(self, e: ft.DragEndEvent):
        self.page.update()

    def open_dialog(self, e):
        self.page.overlay.append(self.dialog)
        self.dialog.open = True
        self.page.update()
    
    def prev_page(self, e):
        if self.page_index > 0:
            self.page_index -= 1
        else:
            self.page_index = len(self.inputpathlist) - 1
        self.img.src = self.inputpathlist[self.page_index]
        self.imgzm.src=self.inputpathlist[self.page_index]
        self.page.update()

    def next_page(self, e):
        if self.page_index < len(self.inputpathlist) - 1:
            self.page_index += 1
        else:
            self.page_index = 0
        self.img.src = self.inputpathlist[self.page_index]
        self.imgzm.src=self.inputpathlist[self.page_index]
        self.page.update()

    def crop_region(self, e):
        pilimg=Image.open(self.img.src)
        pilimg=pilimg.convert('RGB')
        rwidth,rheight=pilimg.size
        if rheight<rwidth:
            window_h=self.dialog_height*rheight/rwidth
            window_w=self.dialog_width
            offset_h=(window_w-window_h)/2
            offset_w=0
        else:
            window_h=self.dialog_height
            window_w=self.dialog_width*rwidth/rheight
            offset_w=(window_h-window_w)/2
            offset_h=0
        hratio=rheight/window_h
        wratio=rwidth/window_w
        cropx=int((self.selection_box.left-offset_w)*wratio)
        cropy=int((self.selection_box.top-offset_h)*hratio)
        cropw=int(self.selection_box.width*wratio)
        croph=int(self.selection_box.height*hratio)
        if cropx>0 and cropy>0 and cropw>10 and croph>0:
            im_crop = pilimg.crop((cropx, cropy, cropx+cropw, cropy+croph))
        else:
            return
        buff = BytesIO()
        im_crop.save(buff, "png")
        self.crop_image.src_base64=base64.b64encode(buff.getvalue()).decode("utf-8")
        self.outputcroppedpath=os.path.join(os.getcwd(),PDFTMPPATH,os.path.basename(self.image_src).split(".")[0]+"_cropped_{}.jpg".format(self.cnt))
        self.mini_ocr(im_crop)
        self.cnt+=1
        self.page.update()
    
    def mini_ocr(self,im_crop):
        self.cropocr_btn.disabled=True
        self.page.update()
        inputname=os.path.basename(self.outputcroppedpath)
        
        tatelinecnt=0
        alllinecnt=0
        self.crop_image.src=im_crop
        npimg = np.array(im_crop)
        img_h,img_w=npimg.shape[:2]
        
        detections,classeslist=process_detector(detector=self.detector,inputname=inputname,npimage=npimg,outputpath=self.outputdirpath,issaveimg=False)
        
        resultobj=[dict(),dict()]
        resultobj[0][0]=list()
        for i in range(17):
            resultobj[1][i]=[]
        for det in detections:
            xmin,ymin,xmax,ymax=det["box"]
            conf=det["confidence"]
            if det["class_index"]==0:
                resultobj[0][0].append([xmin,ymin,xmax,ymax])
            resultobj[1][det["class_index"]].append([xmin,ymin,xmax,ymax,conf])
        xmlstr=convert_to_xml_string3(img_w, img_h, inputname, classeslist, resultobj)
        xmlstr="<OCRDATASET>"+xmlstr+"</OCRDATASET>"

        root = ET.fromstring(xmlstr)
        eval_xml(root, logger=None)
        alllineobj=[]
        alltextlist=[]
        for idx,lineobj in enumerate(root.findall(".//LINE")):
            xmin=int(lineobj.get("X"))
            ymin=int(lineobj.get("Y"))
            line_w=int(lineobj.get("WIDTH"))
            line_h=int(lineobj.get("HEIGHT"))
            try:
                pred_char_cnt=float(lineobj.get("PRED_CHAR_CNT"))
            except:
                pred_char_cnt=100.0
            if line_h>line_w:
                tatelinecnt+=1
            alllinecnt+=1
            lineimg=npimg[ymin:ymin+line_h,xmin:xmin+line_w,:]
            linerecogobj = RecogLine(lineimg,idx,pred_char_cnt)
            alllineobj.append(linerecogobj)

        resultlines=process_cascade(alllineobj,self.recognizer30,self.recognizer50,self.recognizer100,is_cascade=True)
        resultlines=list(resultlines)
        for idx,lineobj in enumerate(root.findall(".//LINE")):
            lineobj.set("STRING",resultlines[idx])
            xmin=int(lineobj.get("X"))
            ymin=int(lineobj.get("Y"))
            line_w=int(lineobj.get("WIDTH"))
            line_h=int(lineobj.get("HEIGHT"))
            try:
                conf=float(lineobj.get("CONF"))
            except:
                conf=0
        patterns = ["□□□", "、、、"]
        resultlines = [s for s in resultlines if not any(p in s for p in patterns)]
        alltextlist.append("\n".join(resultlines))
        if alllinecnt==0 or tatelinecnt/alllinecnt>0.5:
            alltextlist=alltextlist[::-1]
        if self.outputdirpath:
            with open(os.path.join(self.outputdirpath,os.path.basename(inputname).split(".")[0]+".txt"),"w",encoding="utf-8") as wtf:
                wtf.write("\n".join(alltextlist))
        self.resulttext.value="\n".join(alltextlist)
        self.cropocr_btn.disabled=False
        self.open_result_page()
        self.page.update()

    def close_dialog(self, e):
        self.dialog.open=False
        self.page.update()


class CaptureTool:
    def __init__(self, page: ft.Page,config_obj:Dict, detector=None, recognizer30=None, recognizer50=None, recognizer100=None, width: int = 400, height: int = 400):
        self.page = page
        self.config_obj=config_obj
        self.langcode=config_obj["langcode"]
        self.detector = detector
        self.recognizer30 = recognizer30
        self.recognizer50 = recognizer50
        self.recognizer100 = recognizer100
        self.dialog_width = width
        self.dialog_height = height
        self.im_crop=None
        self.img_str=""
        self.result_jsonstr=""
        self.outputdirpath = os.getcwd()

        self.scale_factor = get_windows_scale_factor() 
        self.start_x = 0
        self.start_y = 0
        self.current_x = 0
        self.current_y = 0

        self.original_width = 0
        self.original_height = 0
        self.original_left = 0
        self.original_top = 0
        self.original_bgcolor = None

        self.selection_box = ft.Container(
            border=ft.border.all(2, ft.Colors.RED),
            bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.RED),
            visible=False,
        )
        self.img_control = ft.Image(
            src_base64=None,
            src=None,
            width=self.dialog_width,
            height=self.dialog_height,
            fit=ft.ImageFit.CONTAIN,
            gapless_playback=True 
        )
        self.retry_btn=ft.ElevatedButton(TRANSLATIONS["capturetool_retry_btn"][self.langcode], on_click=self.start_capture)
        self.cboc_fixed = ft.Checkbox(label=TRANSLATIONS["capturetool_fixregion"][self.langcode], value=False,disabled=True)
        self.ocr_btn = ft.ElevatedButton(TRANSLATIONS["capturetool_ocr_button"][self.langcode], on_click=self.mini_ocr)
        
        self.errorlog=ft.Text("")
        self.dialog_content = ft.Column(
            controls=[
                self.errorlog,
                ft.Container(
                    content=self.img_control,
                    border=ft.border.all(1, ft.Colors.GREY),
                    alignment=ft.alignment.center
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True
        )

        self.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(TRANSLATIONS["capturetool_result_title"][self.langcode]),
            content=self.dialog_content,
            actions=[
                ft.Row([self.retry_btn,
                self.cboc_fixed,
                self.ocr_btn,
                ft.ElevatedButton(TRANSLATIONS["common_close"][self.langcode], on_click=self.close_dialog)])
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.resulttext = ft.Text(value="", selectable=True, color=ft.Colors.BLACK)
        self.resultsmessage=ft.Text(value="", selectable=True, color=ft.Colors.BLACK)
        self.llmstatus_text = ft.Text(value="", selectable=True, color=ft.Colors.BLACK)
        self.result_crop_image = ft.Image(src="", width=300, height=300, fit=ft.ImageFit.CONTAIN)
        
        self.crop_image_int = ft.InteractiveViewer(
            min_scale=1,
            max_scale=5,
            boundary_margin=ft.margin.all(20),
            content=ft.Column([self.result_crop_image], width=300, height=300)
        )
        
        self.result_text_col = ft.Column(
            controls=[self.resulttext],
            scroll=ft.ScrollMode.ALWAYS,
            width=600,
            height=300,
        )
        self.result_dialog = ft.AlertDialog(
            title=ft.Text(TRANSLATIONS["capturetool_resultocr_title"][self.langcode]),
            modal=True,
            content=ft.Row(
                controls=[self.crop_image_int, self.result_text_col],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.START
            ),
            actions=[
                self.resultsmessage,
                self.llmstatus_text,
                ft.ElevatedButton(TRANSLATIONS["common_close"][self.langcode], on_click=self.close_result_page)
            ]
        )
        self.bibinfo_dialog = ft.AlertDialog(
            title=ft.Text("書誌情報"),
            content="",
            actions=[
                ft.TextButton("閉じる", on_click=self.close_bibinfo_page)
            ],
        )
        self.overlay_stack = ft.Stack(
            controls=[
                ft.GestureDetector(
                    on_pan_start=self._on_pan_start,
                    on_pan_update=self._on_pan_update,
                    on_pan_end=self._on_pan_end,
                    drag_interval=10,
                ),
                self.selection_box,
            ],
            expand=True,
            visible=False, 
        )

        self.page.overlay.append(self.overlay_stack)


    def start_capture(self, e=None):
        if self.dialog.open:
            self.close_dialog(e)
        if self.cboc_fixed.value:
            self._capture_and_restore(self.x1_phys, self.y1_phys, self.x2_phys, self.y2_phys)
            return
        self.scale_factor = get_windows_scale_factor()
        self.original_width = self.page.window.width
        self.original_height = self.page.window.height
        self.original_left = self.page.window.left
        self.original_top = self.page.window.top
        self.original_bgcolor = self.page.bgcolor

        self.page.window.maximized = True
        self.page.window.title_bar_hidden = True
        self.page.window.title_bar_buttons_hidden = True
        
        self.page.window.always_on_top = True
        self.page.window.opacity = 0.3
        self.page.window.bgcolor = ft.Colors.TRANSPARENT
        self.page.bgcolor = ft.Colors.with_opacity(0.3, ft.Colors.BLACK) 
        
        self.overlay_stack.visible = True
        self.page.update()

    def _on_pan_start(self, e: ft.DragStartEvent):
        self.start_x = e.local_x
        self.start_y = e.local_y
        self.selection_box.visible = True
        self.selection_box.left = self.start_x
        self.selection_box.top = self.start_y
        self.selection_box.width = 0
        self.selection_box.height = 0
        self.page.update()

    def _on_pan_update(self, e: ft.DragUpdateEvent):
        self.current_x = e.local_x
        self.current_y = e.local_y

        left = min(self.start_x, self.current_x)
        top = min(self.start_y, self.current_y)
        width = abs(self.current_x - self.start_x)
        height = abs(self.current_y - self.start_y)

        self.selection_box.left = left
        self.selection_box.top = top
        self.selection_box.width = width
        self.selection_box.height = height
        self.page.update()

    def _on_pan_end(self, e: ft.DragEndEvent):
        x1_local = min(self.start_x, self.current_x)
        y1_local = min(self.start_y, self.current_y)
        x2_local = max(self.start_x, self.current_x)
        y2_local = max(self.start_y, self.current_y)

        offset_x = self.page.window.left or 0
        offset_y = self.page.window.top or 0
        
        x1_global = x1_local + offset_x
        y1_global = y1_local + offset_y
        x2_global = x2_local + offset_x
        y2_global = y2_local + offset_y

        self.x1_phys = int(x1_global * self.scale_factor)
        self.y1_phys = int(y1_global * self.scale_factor)
        self.x2_phys = int(x2_global * self.scale_factor)
        self.y2_phys = int(y2_global * self.scale_factor)
        
        self._capture_and_restore(self.x1_phys, self.y1_phys, self.x2_phys, self.y2_phys)

    def _capture_and_restore(self, x1, y1, x2, y2):
        self.page.window.opacity = 0
        self.page.update()
        
        time.sleep(0.2)

        if (x2 - x1) > 5 and (y2 - y1) > 5:
            try:
                self.im_crop = ImageGrab.grab(bbox=(x1, y1, x2, y2)).convert("RGB")
                self.cboc_fixed.disabled=False
                buffered = io.BytesIO()
                self.im_crop.save(buffered, format="png")
                self.img_control.src_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                self.img_control.src = None
                self.result_crop_image.src_base64=self.img_control.src_base64
                self.page.open(self.dialog)
            except Exception as ex:
                print(f"Capture failed: {ex}")

        self.overlay_stack.visible = False
        self.selection_box.visible = False
        
        self.page.window.opacity = 1
        self.page.window.maximized = False
        self.page.window.title_bar_hidden = False
        self.page.window.title_bar_buttons_hidden = False
        self.page.window.always_on_top = False
        self.page.window.bgcolor = ft.Colors.WHITE
        self.page.bgcolor = self.original_bgcolor
        self.page.update()
        time.sleep(0.2)

        self.page.window.width = self.original_width
        self.page.window.height = self.original_height
        self.page.window.left = self.original_left
        self.page.window.top = self.original_top
        
        self.page.update()

    def mini_ocr(self,e):
        if self.im_crop is None:
            return
        self.ocr_btn.disabled = True
        self.resultsmessage.value=""
        self.page.update()
        try:
            allstart=time.time()
            filename_base = "captureimg"
            pdf_tmp_path = getattr(globals(), "PDFTMPPATH", "tmp")
            self.outputcroppedpath = os.path.join(self.outputdirpath, pdf_tmp_path, f"{filename_base}.jpg")
            tatelinecnt = 0
            alllinecnt = 0

            npimg = np.array(self.im_crop)
            img_h, img_w = npimg.shape[:2]
            
            detections, classeslist = process_detector(
                detector=self.detector,
                inputname=filename_base,
                npimage=npimg,
                outputpath=self.outputdirpath,
                issaveimg=False
            )

            resultobj = [dict(), dict()]
            resultobj[0][0] = list()
            for i in range(17):
                resultobj[1][i] = []
                
            for det in detections:
                xmin, ymin, xmax, ymax = det["box"]
                conf = det["confidence"]
                if det["class_index"] == 0:
                    resultobj[0][0].append([xmin, ymin, xmax, ymax])
                resultobj[1][det["class_index"]].append([xmin, ymin, xmax, ymax, conf])

            xmlstr = convert_to_xml_string3(
                img_w, img_h, filename_base, classeslist, resultobj)
            xmlstr = "<OCRDATASET>" + xmlstr + "</OCRDATASET>"
            root = ET.fromstring(xmlstr)
            
            eval_xml(root, logger=None)

            alllineobj = []
            alltextlist = []

            for idx, lineobj in enumerate(root.findall(".//LINE")):
                xmin = int(lineobj.get("X"))
                ymin = int(lineobj.get("Y"))
                line_w = int(lineobj.get("WIDTH"))
                line_h = int(lineobj.get("HEIGHT"))
                try:
                    pred_char_cnt = float(lineobj.get("PRED_CHAR_CNT"))
                except:
                    pred_char_cnt = 100.0
                
                if line_h > line_w:
                    tatelinecnt += 1
                alllinecnt += 1

                lineimg = npimg[ymin:ymin+line_h, xmin:xmin+line_w, :]
                linerecogobj = RecogLine(lineimg, idx, pred_char_cnt)
                
                alllineobj.append(linerecogobj)

            resultlinesall = process_cascade(
                alllineobj, self.recognizer30, self.recognizer50, self.recognizer100, is_cascade=True
            )
            resultlinesall = list(resultlinesall)
            resjsonarray=[]
            for idx,lineobj in enumerate(root.findall(".//LINE")):
                lineobj.set("STRING",resultlinesall[idx])
                xmin=int(lineobj.get("X"))
                ymin=int(lineobj.get("Y"))
                line_w=int(lineobj.get("WIDTH"))
                line_h=int(lineobj.get("HEIGHT"))
                try:
                    conf=float(lineobj.get("CONF"))
                except:
                    conf=0
                jsonobj={"boundingBox": [[xmin,ymin],[xmin,ymin+line_h],[xmin+line_w,ymin],[xmin+line_w,ymin+line_h]],
                    "id": idx,"isVertical": "true","text": resultlinesall[idx],"isTextline": "true","confidence": conf}
                resjsonarray.append(jsonobj)
            patterns = ["□□□", "、、、"]
            resultlinesall = [s for s in resultlinesall if not any(p in s for p in patterns)]
            alltextlist.append("\n".join(resultlinesall))
            if alllinecnt == 0 or tatelinecnt/alllinecnt > 0.5:
                alltextlist = alltextlist[::-1] 
            final_text = "\n".join(alltextlist)
            self.resultsmessage.value="{:.2f} sec".format(time.time()-allstart)
            self.resulttext.value = final_text
            self.result_jsonstr=json.dumps(resjsonarray,ensure_ascii=False)
            self.open_result_page()

        except Exception as e:
            print(f"OCR Error: {e}")
            self.resulttext.value = f"エラーが発生しました: {e}"
            self.open_result_page()
        finally:
            self.ocr_btn.disabled = False
            self.page.update()
    def open_dialog(self, e=None):
        self.start_capture()
        self.page.overlay.append(self.dialog)
        self.dialog.open = True
        self.page.update()

    def close_dialog(self, e):
        self.dialog.open = False
        self.page.update()

    def open_result_page(self):
        self.dialog.open = False
        self.page.overlay.append(self.result_dialog)
        self.result_dialog.open = True
        self.page.update()

    def close_result_page(self, e):
        self.result_dialog.open = False
        self.dialog.open = True
        self.page.update()

    def open_bibdlg_page(self,content):
        self.result_dialog.open = False
        self.bibinfo_dialog.open = True
        self.page.update()

    def close_bibinfo_page(self, e):
        self.bibinfo_dialog.open=False
        self.result_dialog.open = True
        self.page.update()
    
    def save_config(self):
        with open('userconf.yaml','w',encoding='utf-8')as wf:
            yaml.dump(self.config_obj, wf, default_flow_style=False, allow_unicode=True)


def main(page: ft.Page):
    # デフォルトの重みファイルパスを設定（環境に合わせて調整）
    default_det_weights = "./src/model/rtmdet-s-1280x1280.onnx"
    default_rec_weights = "./src/model/parseq-ndl-32x384-tiny-10.onnx"

    parser = argparse.ArgumentParser(description="Argument for Inference using ONNXRuntime")
    parser.add_argument("--det-weights", type=str, required=False, help="Path to rtmdet onnx file", default=default_det_weights)
    parser.add_argument("--det-classes", type=str, required=False, help="Path to list of class in yaml file",default="./src/config/ndl.yaml")
    parser.add_argument("--det-score-threshold", type=float, required=False, default=0.2)
    parser.add_argument("--det-conf-threshold", type=float, required=False, default=0.25)
    parser.add_argument("--det-iou-threshold", type=float, required=False, default=0.2)

    # カスケード用モデルパス（存在しない場合はデフォルトモデルを使用するようにフォールバック可能だが、ここでは引数として定義）
    # 注意: 環境にこれらのファイルがない場合、起動時にエラーになる可能性があるため、デフォルト値を既存のファイルに合わせます。
    parser.add_argument("--rec-weights30", type=str, required=False, help="Path to parseq-tiny onnx file", default=default_rec_weights)
    parser.add_argument("--rec-weights50", type=str, required=False, help="Path to parseq-tiny onnx file", default=default_rec_weights)
    parser.add_argument("--rec-weights", type=str, required=False, help="Path to parseq-tiny onnx file", default=default_rec_weights)
    
    parser.add_argument("--rec-classes", type=str, required=False, help="Path to list of class in yaml file", default="./src/config/NDLmoji.yaml")
    parser.add_argument("--device", type=str, required=False, help="Device use (cpu or cuda)", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    
    page.title = "NDLkotenOCR-Lite-GUI"
    page.window.icon=os.path.join("assets","icon.png")
    page.window.width = 1024
    page.window.height = 900
    page.window.min_width = 1024
    page.window.min_height = 900

    default_config ={"langcode":"ja",
                     "json":True,
                     "xml":True,
                     "tei":True,
                     "txt":True,
                     "pdf":False,
                     "pdf_viztxt":False,
                     "selected_output_path":None,
                     "prompt":""
                     }
    load_obj={}
    if os.path.exists("userconf.yaml"):
        with open('userconf.yaml', encoding='utf-8')as f:
            load_obj= yaml.safe_load(f)
        if load_obj is None:
            load_obj={}

    config_obj=default_config|load_obj

    page.locale_configuration = ft.LocaleConfiguration(
        supported_locales=[
            ft.Locale("ja", "JP"),
            ft.Locale("en", "US")
        ], 
        current_locale=ft.Locale("ja", "JP") if config_obj["langcode"]=="ja" else ft.Locale("en", "US")
    )
    def save_config():
        with open('userconf.yaml','w',encoding='utf-8')as wf:
            yaml.dump(config_obj, wf, default_flow_style=False, allow_unicode=True)
    
    def handle_locale_change(e):
        index = e.control.selected_index
        if index == 0:
            page.locale_configuration.current_locale = ft.Locale("ja", "JP")
        elif index == 1:
            page.locale_configuration.current_locale = ft.Locale("en", "US")
        config_obj["langcode"]=page.locale_configuration.current_locale.language_code
        save_config()
        page.update()
        renderui()
    
    # モデルのロード
    origin_detector=ocr.get_detector(args=args)
    origin_recognizer=ocr.get_recognizer(args=args)

    def renderui():
        page.clean()
        page.update()
        inputpathlist=[]
        visualizepathlist=[]
        outputtxtlist=[]

        def create_pdf_func(outputpath:str,img:object,bboxlistobj:dict,viztxtflag:bool):
            import reportlab
            from reportlab.pdfgen import canvas
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.cidfonts import UnicodeCIDFont
            from reportlab.lib.utils import ImageReader
            from reportlab.lib.colors import blue
            
            print((img.shape[1],img.shape[0]))
            c = canvas.Canvas(outputpath, pagesize=(img.shape[1],img.shape[0]))
            pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3', isVertical=True))
            pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5', isVertical=False))
            pilimg_data = io.BytesIO()
            pilimg=Image.fromarray(img)
            pilimg.save(pilimg_data, format='png')
            pilimg_data.seek(0)
            side_out = ImageReader(pilimg_data)
            c.drawImage(side_out,0,0)
            if viztxtflag:
                c.setFillColor(blue)
            else:
                c.setFillColor(blue,alpha=0.0)
            for bboxobj in bboxlistobj:
                bbox=bboxobj["boundingBox"]
                text=bboxobj["text"]
                if abs(bbox[2][0]-bbox[0][0])<abs(bbox[1][1]-bbox[0][1]):
                    x_center=(bbox[0][0]+bbox[2][0])//2
                    y_center=img.shape[0]-bbox[0][1]
                    c.setFont('HeiseiMin-W3', abs(bbox[2][0]-bbox[0][0])*3//4)
                    c.drawString(x_center,y_center, text)
                else:
                    x_center=min(bbox[0][0],bbox[2][0])
                    y_center=img.shape[0]-(bbox[0][1]+bbox[1][1])//2
                    c.setFont('HeiseiKakuGo-W5', abs(bbox[1][1]-bbox[0][1]))
                    c.drawString(x_center,y_center, text)
            c.save()
        

        def parts_control(flag:bool):
            file_upload_btn.disabled=flag
            directory_upload_btn.disabled=flag
            directory_output_btn.disabled=flag
            chkbx_visualize.disabled=flag
            customize_btn.disabled=flag
            preview_prev_btn.disabled=flag
            preview_next_btn.disabled=flag
            ocr_btn.disabled=flag
            crop_btn.disabled=flag
            cap_btn.disabled=flag
            localebutton.disabled=flag
            

        def ocr_button_result(e):
            progressbar.value=0
            outputpath=selected_output_path.value
            nonlocal inputpathlist,outputtxtlist,visualizepathlist,preview_index,args
            nonlocal origin_recognizer
            nonlocal origin_detector

            preview_index=0
            parts_control(True)
            page.update()
            progressmessage.value="Start"
            progressmessage.update()
            try:
                tatelinecnt=0
                alllinecnt=0
                allsum=len(inputpathlist)
                allstart=time.time()
                progressbar.value=0
                progressbar.update()
                outputtxtlist.clear()
                visualizepathlist.clear()
                visualizepathlist=[]
                alljsonobjlist=[]
                for idx,inputpath in enumerate(inputpathlist):
                    progressmessage.value=inputpath
                    progressmessage.update()
                    pil_image = Image.open(inputpath).convert('RGB')
                    npimg = np.array(pil_image)
                    start = time.time()
                    inputdivlist=[]
                    imgnamelist=[]
                    inputdivlist.append(npimg)
                    imgnamelist.append(os.path.basename(inputpath))
                    allxmlstr="<OCRDATASET>\n"
                    alltextlist=[]
                    resjsonarray=[]
                    for img,imgname in zip(inputdivlist,imgnamelist):
                        img_h,img_w=img.shape[:2]
                        # ヘルパー関数を使用して検出を実行
                        detections,classeslist=process_detector(detector=origin_detector,inputname=imgname,npimage=img,outputpath=outputpath,issaveimg=False)
                        e1=time.time()
                        print("layout detection Done!",e1-start)
                        
                        resultobj=[dict(),dict()]
                        resultobj[0][0]=list()
                        for i in range(17):
                            resultobj[1][i]=[]
                        for det in detections:
                            xmin,ymin,xmax,ymax=det["box"]
                            conf=det["confidence"]
                            # pred_char_count キーが存在しない場合のフォールバック
                            char_count = det.get("pred_char_count", 0)
                            if det["class_index"]==0:
                                resultobj[0][0].append([xmin,ymin,xmax,ymax])
                            resultobj[1][det["class_index"]].append([xmin,ymin,xmax,ymax,conf,char_count])
                        
                        xmlstr=convert_to_xml_string3(img_w, img_h, imgname, classeslist, resultobj)
                        xmlstr="<OCRDATASET>"+xmlstr+"</OCRDATASET>"
                        
                        root = ET.fromstring(xmlstr)
                        eval_xml(root, logger=None)
                        alllinerecogobj=[]
                        for idx,lineobj in enumerate(root.findall(".//LINE")):
                            xmin=int(lineobj.get("X"))
                            ymin=int(lineobj.get("Y"))
                            line_w=int(lineobj.get("WIDTH"))
                            line_h=int(lineobj.get("HEIGHT"))
                            try:
                                pred_char_cnt=float(lineobj.get("PRED_CHAR_CNT"))
                            except:
                                pred_char_cnt=0.0
                            if line_h>line_w:
                                tatelinecnt+=1
                            alllinecnt+=1
                            lineimg = img[ymin:ymin+line_h,xmin:xmin+line_w,:]
                            linerecogobj = RecogLine(lineimg,idx,pred_char_cnt)
                            alllinerecogobj.append(linerecogobj)
                        resultlinesall=process_cascade(alllinerecogobj,recognizer30=None,recognizer50=None,recognizer100=origin_recognizer)
                        alltextlist.append("\n".join(resultlinesall))
                        for idx,lineobj in enumerate(root.findall(".//LINE")):
                            lineobj.set("STRING",resultlinesall[idx])
                            xmin=int(lineobj.get("X"))
                            ymin=int(lineobj.get("Y"))
                            line_w=int(lineobj.get("WIDTH"))
                            line_h=int(lineobj.get("HEIGHT"))
                            try:
                                conf=float(lineobj.get("CONF"))
                            except:
                                conf=0
                            jsonobj={"boundingBox": [[xmin,ymin],[xmin,ymin+line_h],[xmin+line_w,ymin],[xmin+line_w,ymin+line_h]],
                                "id": idx,"isVertical": "true","text": resultlinesall[idx],"isTextline": "true","confidence": conf}
                            resjsonarray.append(jsonobj)
                        allxmlstr+=(ET.tostring(root.find("PAGE"), encoding='unicode')+"\n")
                        e2=time.time()
                    allxmlstr+="</OCRDATASET>"
                    if alllinecnt==0 or tatelinecnt/alllinecnt>0.5:
                        alltextlist=alltextlist[::-1]
                    outputtxtlist.append("\n".join(alltextlist))
                    alljsonobj={
                        "contents":[resjsonarray],
                        "imginfo": {
                            "img_width": img_w,
                            "img_height": img_h,
                            "img_path":inputpath,
                            "img_name":os.path.basename(inputpath)
                        }
                    }
                    alljsonobjlist.append(alljsonobj)
                    if chkbx_xml.value:
                        with open(os.path.join(outputpath,os.path.basename(inputpath).split(".")[0]+".xml"),"w",encoding="utf-8") as wf:
                            wf.write(allxmlstr)
                    if chkbx_visualize.value:
                        output_vizpath=os.path.join(outputpath,"viz_"+os.path.basename(inputpath))
                        if output_vizpath.split(".")[-1]=="jp2":
                            output_vizpath=output_vizpath[:-4]+".jpg"
                        visualizepathlist.append(output_vizpath)
                        # Detectorのdrawメソッドを呼び出し
                        process_detector(detector=origin_detector,inputname=os.path.basename(inputpath),npimage=npimg,outputpath=outputpath,issaveimg=True)

                    if chkbx_json.value:
                        with open(os.path.join(outputpath,os.path.basename(inputpath).split(".")[0]+".json"),"w",encoding="utf-8") as wf:
                            wf.write(json.dumps(alljsonobj,ensure_ascii=False,indent=2))
                    if chkbx_txt.value:
                        with open(os.path.join(outputpath,os.path.basename(inputpath).split(".")[0]+".txt"),"w",encoding="utf-8") as wtf:
                            wtf.write("\n".join(alltextlist))
                    if chkbx_pdf.value:
                        create_pdf_func(os.path.join(outputpath,os.path.basename(inputpath).split(".")[0]+".pdf"),img,resjsonarray,chkbx_pdf_viztxt.value)
                        
                    progressbar.value+=1/allsum
                    preview_prev_btn.disabled=False
                    preview_next_btn.disabled=False
                    preview_text.value= outputtxtlist[preview_index]
                    if len(visualizepathlist)>0:
                        preview_image.src = visualizepathlist[preview_index]
                        current_visualizeimgname.value=os.path.basename(inputpathlist[preview_index])
                    else:
                        preview_image.src = inputpathlist[preview_index]
                        current_visualizeimgname.value=os.path.basename(inputpathlist[preview_index])
                    preview_image.update()
                    page.update()
                if config_obj["langcode"]=="ja":
                    progressmessage.value="{} 画像OCR完了 / 所要時間 {:.2f} 秒".format(allsum,time.time()-allstart)
                else:
                    progressmessage.value="{} images completed / Total time {:.2f} sec".format(allsum,time.time()-allstart)
                progressmessage.update()
                if chkbx_tei.value:
                    with open(os.path.join(outputpath,os.path.basename(inputpathlist[0]).split(".")[0]+"_tei.xml"),"wb") as wf:
                        allxmlstrtei=convert_tei(alljsonobjlist)
                        wf.write(allxmlstrtei)
            except Exception as e:
                print(e)
                progressmessage.value=str(e)
                progressmessage.update()
            finally:
                parts_control(False)
                page.update()

        
        def pick_files_result(e: ft.FilePickerResultEvent):
            if e.files:
                selected_input_path.value=e.files[0].path
                nonlocal inputpathlist,outputtxtlist
                inputpathlist.clear()
                outputtxtlist.clear()
                ext=e.files[0].path.split(".")[-1]
                if ext=="pdf":
                    filestem=os.path.basename(e.files[0].path)[:-4]
                    if config_obj["langcode"]=="ja":
                        progressmessage.value="pdfファイルの前処理中…… {} ".format(e.files[0].path)
                    else:
                        progressmessage.value="preprocessing pdf…… {} ".format(e.files[0].path)
                    parts_control(True)
                    page.update()
                    for p in glob.glob(os.path.join(os.getcwd(),PDFTMPPATH,"*.jpg")):
                        if os.path.isfile(p):
                            os.remove(p)
                    os.makedirs(os.path.join(os.getcwd(),PDFTMPPATH), exist_ok=True)
                    doc = pypdfium2.PdfDocument(selected_input_path.value)
                    pdfarray=doc.render(pypdfium2.PdfBitmap.to_pil,
                                            page_indices = [i for i in range(len(doc))],
                                            scale = 100/72)
                    for ix,image in enumerate(list(pdfarray)):
                        outputtmppath=os.path.join(os.getcwd(),PDFTMPPATH,"{}_{:05}.jpg".format(filestem,ix))
                        inputpathlist.append(outputtmppath)
                        image=image.convert("RGB")
                        image.save(outputtmppath)
                    if config_obj["langcode"]=="ja":
                        progressmessage.value="pdfファイルの前処理完了"
                    else:
                        progressmessage.value="Preprocessing of pdf complete"
                    parts_control(False)
                    ocr_btn.disabled=True
                    crop_btn.disabled=True
                    page.update()
                else:
                    inputpathlist.append(e.files[0].path)
                selector.set_image(inputpathlist)
                if selected_output_path.value!=None:
                    parts_control(False)
            selected_input_path.update()
            page.update()

        def pick_directory_result(e: ft.FilePickerResultEvent):
            if e.path:
                selected_input_path.value = e.path
                nonlocal inputpathlist,outputtxtlist
                inputpathlist.clear()
                outputtxtlist.clear()
                cleanflag=False
                for inputname in os.listdir(e.path):
                    inputpath=os.path.join(e.path,inputname)
                    ext=inputpath.split(".")[-1]
                    if ext.lower() in ["jpg","png","tiff","jp2","tif","jpeg","bmp"] and os.path.isfile(inputpath):
                        inputpathlist.append(inputpath)
                    elif ext=="pdf" and os.path.isfile(inputpath):
                        filestem=os.path.basename(inputpath)[:-4]
                        if config_obj["langcode"]=="ja":
                            progressmessage.value="pdfファイルの前処理中…… {} ".format(e.files[0].path)
                        else:
                            progressmessage.value="preprocessing pdf…… {} ".format(e.files[0].path)
                        parts_control(True)
                        page.update()
                        if not cleanflag:
                            for p in glob.glob(os.path.join(os.getcwd(),PDFTMPPATH,"*.jpg")):
                                if os.path.isfile(p):
                                    os.remove(p)
                            cleanflag=True
                        os.makedirs(os.path.join(os.getcwd(),PDFTMPPATH), exist_ok=True)
                        doc = pypdfium2.PdfDocument(inputpath)
                        pdfarray=doc.render(pypdfium2.PdfBitmap.to_pil,
                                                page_indices = [i for i in range(len(doc))],
                                                scale = 100/72)
                        for ix,image in enumerate(list(pdfarray)):
                            outputtmppath=os.path.join(os.getcwd(),PDFTMPPATH,"{}_{:05}.jpg".format(filestem,ix))
                            inputpathlist.append(outputtmppath)
                            image=image.convert("RGB")
                            image.save(outputtmppath)

                        if config_obj["langcode"]=="ja":
                            progressmessage.value="pdfファイルの前処理完了"
                        else:
                            progressmessage.value="Preprocessing of pdf complete"
                        parts_control(False)
                        crop_btn.disabled=True
                        ocr_btn.disabled=True
                        page.update()
                selector.set_image(inputpathlist)
            if selected_output_path.value!=None and len(inputpathlist)>0:
                parts_control(False)
            selected_input_path.update()
            page.update()

        def pick_output_result(e: ft.FilePickerResultEvent):
            nonlocal inputpathlist
            if e.path:
                selected_output_path.value = e.path
                selected_output_path.update()
                config_obj["selected_output_path"]=e.path
                save_config()
                selector.set_outputdir(e.path)
                if len(inputpathlist)>0:
                    parts_control(False)
            page.update()

        preview_index=0
        def next_image(e):
            nonlocal inputpathlist,outputtxtlist,preview_index
            if preview_index < min(len(inputpathlist) - 1,len(outputtxtlist) - 1):
                preview_index += 1
            else:
                preview_index = 0

            if len(visualizepathlist)>0:
                preview_image.src = visualizepathlist[preview_index]
                current_visualizeimgname.value=os.path.basename(visualizepathlist[preview_index])
            elif 0<=preview_index<len(outputtxtlist):
                preview_image.src = inputpathlist[preview_index]
                current_visualizeimgname.value=os.path.basename(inputpathlist[preview_index])
            if 0<=preview_index<len(outputtxtlist):
                preview_text.value=outputtxtlist[preview_index]
            preview_image.update()
            preview_text.update()
            page.update()


        def prev_image(e):
            nonlocal inputpathlist,outputtxtlist,preview_index
            if preview_index > 0:
                preview_index -= 1
            else:
                preview_index = min(len(inputpathlist) - 1,len(outputtxtlist) - 1)
            
            if len(visualizepathlist)>0:
                preview_image.src = visualizepathlist[preview_index]
                current_visualizeimgname.value=os.path.basename(visualizepathlist[preview_index])
            elif 0<=preview_index<len(outputtxtlist):
                preview_image.src = inputpathlist[preview_index]
                current_visualizeimgname.value=os.path.basename(inputpathlist[preview_index])
            if 0<=preview_index<len(outputtxtlist):
                preview_text.value=outputtxtlist[preview_index]
            preview_image.update()
            preview_text.update()
            page.update()
        

        def handle_customize_dlg_modal_close(e):
            config_obj.update({
                "json":chkbx_json.value,
                "txt":chkbx_txt.value,
                "xml":chkbx_xml.value,
                "tei":chkbx_tei.value,
                "pdf":chkbx_pdf.value,
                "pdf_viztxt":chkbx_pdf_viztxt.value,
            })
            save_config()
            page.close(customize_dlg_modal)
        
        def change_pdfstatus(e):
            chkbx_pdf_viztxt.disabled=not chkbx_pdf.value
            chkbx_pdf_viztxt.update()
        

        preview_image=ft.Image(src="dummy.dat", width=400, height=300,gapless_playback=True)
        preview_text=ft.Text(value="",height=300,selectable=True)

        pick_directory_dialog = ft.FilePicker(on_result=pick_directory_result)
        pick_output_dialog = ft.FilePicker(on_result=pick_output_result)
        pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
        progressbar = ft.ProgressBar(width=400,value=0)
        selected_input_path = ft.Text(selectable=True)
        selected_output_path = ft.Text(config_obj["selected_output_path"],selectable=True)
        current_visualizeimgname=ft.Text(selectable=True)
        progressmessage=ft.Text()
        chkbx_visualize = ft.Checkbox(label=TRANSLATIONS["main_visualize_label"][config_obj["langcode"]], value=True)
        chkbx_json = ft.Checkbox(label="JSON形式", value=config_obj["json"])
        chkbx_txt = ft.Checkbox(label="TXT形式", value=config_obj["txt"])
        chkbx_xml = ft.Checkbox(label="XML形式", value=config_obj["xml"])
        chkbx_tei = ft.Checkbox(label="TEI形式", value=config_obj["tei"])
        chkbx_pdf = ft.Checkbox(label="透明テキスト付PDF(ベータ)", value=config_obj["pdf"],on_change=change_pdfstatus)
        chkbx_pdf_viztxt = ft.Checkbox(label="PDFに青色で文字を重ねる", value=config_obj["pdf_viztxt"],disabled=not chkbx_pdf.value)

        
        file_upload_btn=ft.ElevatedButton(
                        TRANSLATIONS["main_file_upload_btn"][config_obj["langcode"]],
                        icon=ft.Icons.UPLOAD_FILE,
                        on_click=lambda _: pick_files_dialog.pick_files(
                            allow_multiple=False
                        ),
                    )
        directory_upload_btn=ft.ElevatedButton(
                        TRANSLATIONS["main_directory_upload_btn"][config_obj["langcode"]],
                        icon=ft.Icons.FOLDER_OPEN,
                        on_click=lambda _: pick_directory_dialog.get_directory_path(),
                    )
        directory_output_btn=ft.ElevatedButton(
                        TRANSLATIONS["main_directory_output_btn"][config_obj["langcode"]],
                        on_click=lambda _: pick_output_dialog.get_directory_path(),
                    )
        ocr_btn=ft.ElevatedButton(text="OCR",
                                    on_click=ocr_button_result,
                                    style=ft.ButtonStyle(
                                        padding=30,
                                        shape=ft.RoundedRectangleBorder(radius=10)),
                                    disabled=True)
        preview_image_col = ft.Column(
            controls=[preview_image],
            width=400,
            height=300,
            expand=False
        )
        
        preview_image_int=ft.InteractiveViewer(
                min_scale=1,
                max_scale=10,
                boundary_margin=ft.margin.all(20),
                content=preview_image_col,
        )
        preview_text_col = ft.Column(
            controls=[preview_text],
            scroll=ft.ScrollMode.ALWAYS,
            width=600,
            height=300,
            expand=False
        )
        preview_prev_btn=ft.ElevatedButton(text=TRANSLATIONS["main_prev_btn"][config_obj["langcode"]], on_click=prev_image,disabled=True)
        preview_next_btn=ft.ElevatedButton(text=TRANSLATIONS["main_next_btn"][config_obj["langcode"]], on_click=next_image,disabled=True)
        customize_btn=ft.ElevatedButton(TRANSLATIONS["main_customize_btn"][config_obj["langcode"]], on_click=lambda e: page.open(customize_dlg_modal))
        customize_dlg_modal = ft.AlertDialog(
            modal=True,
            title=ft.Text(TRANSLATIONS["customize_dlg_modal_title"][config_obj["langcode"]]),
            content=ft.Text(TRANSLATIONS["customize_dlg_modal_explain"][config_obj["langcode"]]),
            actions=[
                chkbx_txt,
                chkbx_json,
                ft.Row([chkbx_xml,chkbx_tei]),
                ft.Row([chkbx_pdf,chkbx_pdf_viztxt]),
                ft.TextButton("OK", on_click=handle_customize_dlg_modal_close),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        selector = ImageSelector(page,config_obj,detector=origin_detector,
                                recognizer100=origin_recognizer,
                                outputdirpath=selected_output_path.value)

        capture_tool=CaptureTool(page,config_obj,detector=origin_detector,
                                recognizer100=origin_recognizer)
        page.overlay.extend([customize_dlg_modal,pick_files_dialog,pick_directory_dialog,pick_output_dialog,
                            selector.dialog,selector.zoom_dialog,selector.result_dialog,
                            capture_tool.dialog,capture_tool.result_dialog])
        crop_btn = ft.ElevatedButton(text="Crop&OCR",
                                        on_click=selector.open_dialog,
                                        style=ft.ButtonStyle(
                                            padding=10,
                                            shape=ft.RoundedRectangleBorder(radius=10)),
                                        disabled=True)
        cap_btn = ft.ElevatedButton(text=TRANSLATIONS["main_cap_btn"][config_obj["langcode"]],
                                        on_click=capture_tool.start_capture,
                                        style=ft.ButtonStyle(
                                            padding=10,
                                            shape=ft.RoundedRectangleBorder(radius=10)),
                                        disabled=False)
        explain_label=ft.Text(TRANSLATIONS["main_explain"][config_obj["langcode"]])
        localebutton=ft.CupertinoSlidingSegmentedButton(
                        selected_index=0 if config_obj["langcode"]=="ja" else 1,
                        thumb_color=ft.Colors.BLUE_400,
                        on_change=handle_locale_change,
                        controls=[ft.Text("日本語"), ft.Text("English")],
                    )
        page.add(
            ft.Row(
                [
                   localebutton, 
                ]
            ),
            ft.Row(
                [
                    explain_label,
                    cap_btn
                ],
                ),
            ft.Divider(),
            ft.Row(
                [
                    file_upload_btn,
                    directory_upload_btn,
                    ft.Text(TRANSLATIONS["main_target_label"][config_obj["langcode"]]),
                    selected_input_path,
                ]
            ),
            ft.Divider(),
            ft.Row(
                [
                    directory_output_btn,
                    ft.Text(TRANSLATIONS["main_output_label"][config_obj["langcode"]]),
                    selected_output_path,
                ]
            ),
            ft.Divider(),
            ft.Row(
                [ocr_btn,
                crop_btn,
                ft.Column([chkbx_visualize,customize_btn
                            ]),
                ft.Column([progressmessage,progressbar]),
                ]
            ),
            ft.Divider(),
            ft.Row([ft.Text(TRANSLATIONS["main_preview_label"][config_obj["langcode"]]),preview_prev_btn,preview_next_btn,current_visualizeimgname]),
            ft.Row([preview_image_int,preview_text_col])
        )
        page.update()
    renderui()
ft.app(main,assets_dir="assets")