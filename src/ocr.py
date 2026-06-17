import sys
sys.setrecursionlimit(5000)
import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
from rtmdet import RTMDet
from parseq import PARSEQ

from yaml import safe_load
from concurrent.futures import ThreadPoolExecutor
import time
import shutil
import json
import glob
from reading_order.xy_cut.eval import eval_xml
from ndl_parser import convert_to_xml_string3
from reading_order.xy_cut.eval import eval_xml
from tools.ndlkoten2tei import convert_tei

_DETECTOR_CACHE = {}
def get_detector(args):
    _det_key = (args.det_weights, args.det_classes, args.det_score_threshold,
                args.det_conf_threshold, args.det_iou_threshold, args.device)
    if _det_key in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[_det_key]
    weights_path = args.det_weights
    classes_path = args.det_classes
    assert os.path.isfile(weights_path), f"There's no weight file with name {weights_path}"
    assert os.path.isfile(classes_path), f"There's no classes file with name {weights_path}"
    detector = RTMDet(model_path=weights_path,
                      class_mapping_path=classes_path,
                      score_threshold=args.det_score_threshold,
                      conf_thresold=args.det_conf_threshold,
                      iou_threshold=args.det_iou_threshold,
                      device=args.device)
    _DETECTOR_CACHE[_det_key] = detector
    return detector
def get_recognizer(args):
    weights_path = args.rec_weights
    classes_path = args.rec_classes

    assert os.path.isfile(weights_path), f"There's no weight file with name {weights_path}"
    assert os.path.isfile(classes_path), f"There's no classes file with name {weights_path}"

    charobj=None
    with open(classes_path,encoding="utf-8") as f:
        charobj=safe_load(f)
    charlist=list(charobj["model"]["charset_train"])
    
    recognizer = PARSEQ(model_path=weights_path,charlist=charlist,device=args.device)
    return recognizer


def inference_on_detector(args,inputname:str,npimage:np.ndarray,outputpath:str,issaveimg:bool=True):
    print("[INFO] Intialize Model")
    detector = get_detector(args)
    print("[INFO] Inference Image")
    detections = detector.detect(npimage)
    classeslist=list(detector.classes.values())
    drawimage = npimage.copy()
    pil_image =detector.draw_detections(drawimage, detections=detections)
    if issaveimg:
        os.makedirs(outputpath,exist_ok=True)
        output_path = os.path.join(outputpath,f"viz_{Path(inputname).name}")
        print(f"[INFO] Saving result on {output_path}")
        pil_image.save(output_path)
    return detections,classeslist


def process(args):
    rawinputpathlist=[]
    inputpathlist=[]
    if args.sourcedir is not None:
        for inputpath in glob.glob(os.path.join(args.sourcedir,"*")):
            rawinputpathlist.append(inputpath)
    if args.sourceimg is not None:
        rawinputpathlist.append(args.sourceimg)
    
    for inputpath in rawinputpathlist:
        ext=inputpath.split(".")[-1]
        if ext in ["jpg","png","tiff","jp2","tif","jpeg","bmp"]:
            inputpathlist.append(inputpath)
    if len(inputpathlist)==0:
        print("Images are not found.")
        return
    if not os.path.exists(args.output):
        print("Output Directory is not found.")
        return
    print(inputpathlist)
    recognizer=get_recognizer(args=args)
    tatelinecnt=0
    alllinecnt=0
    alljsonobjlist=[]
    for inputpath in inputpathlist:
        ext=inputpath.split(".")[-1]
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
            detections,classeslist=inference_on_detector(args=args,inputname=imgname,npimage=img,outputpath=args.output,issaveimg=args.viz)
            e1=time.time()
            #print("layout detection Done!",e1-start)
            resultobj=[dict(),dict()]
            resultobj[0][0]=list()
            for i in range(16):
                resultobj[1][i]=[]
            for det in detections:
                xmin,ymin,xmax,ymax=det["box"]
                conf=det["confidence"]
                if det["class_index"]==0:
                    resultobj[0][0].append([xmin,ymin,xmax,ymax])
                resultobj[1][det["class_index"]].append([xmin,ymin,xmax,ymax,conf])
            xmlstr=convert_to_xml_string3(img_w, img_h, imgname, classeslist, resultobj,score_thr = 0.3,min_bbox_size= 5,use_block_ad= False)
            xmlstr="<OCRDATASET>"+xmlstr+"</OCRDATASET>"
            root = ET.fromstring(xmlstr)
            eval_xml(root, logger=None)
            targetdflist=[]
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="thread") as executor:
                for lineobj in root.findall(".//LINE"):
                    xmin=int(lineobj.get("X"))
                    ymin=int(lineobj.get("Y"))
                    line_w=int(lineobj.get("WIDTH"))
                    line_h=int(lineobj.get("HEIGHT"))
                    if line_h>line_w:
                        tatelinecnt+=1
                    alllinecnt+=1
                    lineimg=img[ymin:ymin+line_h,xmin:xmin+line_w,:]
                    targetdflist.append(lineimg)
                resultlines = executor.map(recognizer.read, targetdflist)
                resultlines=list(resultlines)
                alltextlist.append("\n".join(resultlines))
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
                    jsonobj={"boundingBox": [[xmin,ymin],[xmin,ymin+line_h],[xmin+line_w,ymin],[xmin+line_w,ymin+line_h]],
                        "id": idx,"isVertical": "true","text": resultlines[idx],"isTextline": "true","confidence": conf}
                    resjsonarray.append(jsonobj)
            allxmlstr+=(ET.tostring(root.find("PAGE"), encoding='unicode')+"\n")
            e2=time.time()
            #print("Text Recogntion Done. Calculation time:",e2-e1)
        allxmlstr+="</OCRDATASET>"
        if tatelinecnt/alllinecnt>0.5:
            alltextlist=alltextlist[::-1]
        
        with open(os.path.join(args.output,os.path.basename(inputpath).split(".")[0]+".xml"),"w",encoding="utf-8") as wf:
            wf.write(allxmlstr)
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
        alljsonstr=json.dumps(alljsonobj,ensure_ascii=False,indent=2)
        with open(os.path.join(args.output,os.path.basename(inputpath).split(".")[0]+".json"),"w",encoding="utf-8") as wf:
            wf.write(alljsonstr)
        with open(os.path.join(args.output,os.path.basename(inputpath).split(".")[0]+".txt"),"w",encoding="utf-8") as wtf:
            wtf.write("\n".join(alltextlist))
        #print("Total calculation time (Detection + Recognition):",time.time()-start)
    with open(os.path.join(args.output,os.path.basename(inputpathlist[0]).split(".")[0]+"_tei.xml"),"wb") as wf:
        allxmlstrtei=convert_tei(alljsonobjlist)
        wf.write(allxmlstrtei)


if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Arguments for NDLkotenOCR-Lite")

    parser.add_argument("--sourcedir", type=str, required=False, help="Path to image directory")
    parser.add_argument("--sourceimg", type=str, required=False, help="Path to image directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--viz", type=bool, required=False, help="Save visualized image",default=False)
    parser.add_argument("--det-weights", type=str, required=False, help="Path to rtmdet onnx file", default="model/rtmdet-s-1280x1280.onnx")
    parser.add_argument("--det-classes", type=str, required=False, help="Path to list of class in yaml file",default="config/ndl.yaml")
    parser.add_argument("--det-score-threshold", type=float, required=False, default=0.3)
    parser.add_argument("--det-conf-threshold", type=float, required=False, default=0.3)
    parser.add_argument("--det-iou-threshold", type=float, required=False, default=0.3)
    parser.add_argument("--rec-weights", type=str, required=False, help="Path to parseq-tiny onnx file", default="model/parseq-ndl-32x384-tiny-10.onnx")
    parser.add_argument("--rec-classes", type=str, required=False, help="Path to list of class in yaml file", default="config/NDLmoji.yaml")
    parser.add_argument("--device", type=str, required=False, help="Device use (cpu or cuda)", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    process(args)
    

