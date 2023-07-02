import json
import os
import tkinter as tk
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)

def get_annotation(mask, image=None):
    eps = 0.0010
    # the findCountours function was originally supploed the RETR_TREE argument instead of the RETR_LIST, the change is mainly experimental
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:

        # Valid polygons have >= 6 coordinates (3 points)
        # print("initial length of contour", contour.size)
        if contour.size >= 6:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            # print("length approximated:", len(approx))
            # cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
            # plt.imshow(image)
            # plt.show()
            segmentation.append(contour.flatten().tolist())
    if len(segmentation) == 0:
        return segmentation, [0, 0, 0, 0], 0.0
    RLEs = cocomask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
    RLE = cocomask.merge(RLEs)
    # RLE = cocomask.encode(np.asfortranarray(mask))
    area = cocomask.area(RLE)
    [x, y, w, h] = cv2.boundingRect(mask)

    # if image is not None:
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     cv2.drawContours(image, contours, -1, (0, 255, 0), -1)
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #     cv2.imwrite(
    #         "/media/likhith/Data/mowito/mmdeploy/mw_mmdetection/testt.png", image
    #     )

    return segmentation, [x, y, w, h], area

class App:
    def __init__(self, master):
        self.master = master
        self.dir_path = ""
        self.image_list = []
        self.current_image_idx = 0
        self.bbox_coords = []
        self.box_coordinates = []
        self.point_of_interest = []
        self.coco_annotation = []
        self.previous_polygons = []
        self.image_path = None
        self.category_id = 1
        self.annotation_id = 1
        self.canvas = tk.Canvas(master, width=1280, height=720)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.select_point)
        self.canvas.bind("<Button-3>", self.run_point_sam)
        self.canvas.bind("<BackSpace>", self.delete_last_polygon)
        self.canvas.focus_set()
        self.load_button = tk.Button(master, text="Load Images", command=self.load_images)
        self.load_button.pack()
        self.next_button = tk.Button(master, text="Next Image", command=self.next_image)
        self.next_button.pack()
        self.save_button = tk.Button(master, text="Save Annotations", command=self.save_annotations)
        self.save_button.pack()
        category_id_label = tk.Label(master, text="Category ID")
        category_id_label.pack(side=tk.LEFT)
        self.category_id_entry = tk.Entry(master)
        self.category_id_entry.pack(side=tk.LEFT)
        self.category_id_entry.bind("<Return>", self.update_category_id)
        self.coco_dict = {
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
            "categories": [{"id": self.category_id, "name": "object", "supercategory": ""}],
            "images": [],
            "annotations": [],
        }

    def update_category_id(self, event):
        self.category_id = int(self.category_id_entry.get())
        print(f"Category ID updated to {self.category_id}")
        self.canvas.focus_set()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Main Window is closing, call any function you'd like here!")


    def load_images(self):
        self.dir_path = filedialog.askdirectory()
        print("self.dir_path",self.dir_path)
        self.image_list = [os.path.join(self.dir_path, f) for f in os.listdir(self.dir_path) if f.endswith(".jpg")]
        print("number of images loaded:",len(self.image_list))
        self.load_image()
        
    def load_image(self):
        self.canvas.delete("all")
        self.image_path  = self.image_list[self.current_image_idx]
        print(self.image_path)
        image = Image.open(self.image_list[self.current_image_idx])
        w, h = image.size
        self.canvas.config(width=w, height=h)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        image_bgr = cv2.imread(self.image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.height, self.width, _ = image_bgr.shape
        print('Generating SAM masks')
        self.mask_predictor = SamPredictor(sam)
        self.mask_predictor.set_image(image_rgb)
        print('Generated successfully')
        
    def select_point(self,event):
        x, y = event.x, event.y
        self.point_of_interest.append([x,y])
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill='red')
    
    def to_coco(self,polygons):
        # Add image information
        image_id = self.current_image_idx
        image_name = os.path.basename(self.image_path)
        image_dict = {
            "id": image_id,
            "width": self.width,
            "height": self.height,
            "file_name": image_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }
        self.coco_dict['images'].append(image_dict)
        # Add annotation information
        for i, polygon in enumerate(polygons):
            mask_image = Image.new("L", (self.width, self.height), 0)
            ImageDraw.Draw(mask_image).polygon(polygon, outline=1, fill=1)
            binary_mask = np.array(mask_image)
            
            segmentation, bbox, area = get_annotation(binary_mask)
            
            
            # coco_polygon = [int(point) for tuple in polygon for point in tuple]
            # rows = np.any(binary_mask, axis=1)
            # cols = np.any(binary_mask, axis=0)
            # y1, y2 = np.where(rows)[0][[0, -1]]
            # x1, x2 = np.where(cols)[0][[0, -1]]

            annotation = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": self.category_id,
                "segmentation": segmentation,
                "area": float(area), #int(np.sum(binary_mask)),
                "bbox": bbox, #[int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "iscrowd": 0,
                "attributes": {"occluded": False}
            }
            # Add annotation to list of annotations
            self.coco_dict["annotations"].append(annotation)
            self.annotation_id +=1

    def save_annotations(self):
        # json_file_path = filedialog.askdirectory()
        json_file_path = "annotations.json"
        with open(json_file_path, "w") as f:
            json.dump(self.coco_dict, f, indent=4)
        print("Annotations saved!")

    def delete_last_polygon(self, event):
        if len(self.previous_polygons) > 0:
            self.previous_polygons.pop()
            self.redraw_image_with_polygons()

    def redraw_image_with_polygons(self):
        print('Removing recent polygon')
        image_path = self.image_list[self.current_image_idx]
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image, mode='RGBA')

        # Draw all the previous polygons on the image
        outline_color = (255, 0, 0)
        fill_color = (0, 255, 0, 50)
        for prev_polygon in self.previous_polygons:
            draw.polygon(prev_polygon, outline=outline_color, fill=fill_color)

        # Display the image with the polygons on the canvas
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def display_image_and_polygon(self,mask):
        image_path = self.image_list[self.current_image_idx]
        mask = mask.astype(np.uint8)
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        # segmentation = []

        # Find the contour of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)

        # segmentation.append(approx.flatten().tolist())

        # RLEs = cocomask.frPyObjects(segmentation,mask.shape[0],mask.shape[1])
        # RLE = cocomask.merge(RLEs)
        # area = cocomask.area(RLE)
        # [x,y,w,h] = cv2.boundingRect(mask)

        # Convert the polygon points to a list of tuples
        polygon = [tuple(point[0]) for point in approx]
        self.previous_polygons.append(polygon)
        # Draw the polygon on the image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image, mode='RGBA') # Use RGBA mode to support transparency
        outline_color = (255, 0, 0) # Red outline
        fill_color = (0, 255, 0, 50) # Semi-transparent light green fill (opacity: 50/255)
        for prev_polygon in self.previous_polygons:
            draw.polygon(prev_polygon, outline=outline_color, fill=fill_color)

        # Display the image with the polygon on the canvas
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def run_point_sam(self,event):
        input_point = np.array(self.point_of_interest)
        input_label = np.array([1]*len(self.point_of_interest))
        try:
            masks, _, _ = self.mask_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
                )
            self.point_of_interest = []
            self.display_image_and_polygon(masks[0])
        except IndexError:
            print("index out of range")
            #self.save_annotations()
            #exit()
        

    def next_image(self):
        self.to_coco(self.previous_polygons)
        print("coco annotation status:")
        print("number of images:",len(self.coco_dict['images']))
        print("total number of anns:",len(self.coco_dict['annotations']))

        self.previous_polygons = []
        self.box_coordinates = []
        self.point_of_interest = []
        self.current_image_idx += 1
        if self.current_image_idx >= len(self.image_list):
            print("looped over all images")
            self.current_image_idx = 0
            self.save_annotations()
            exit()
        else:
            self.load_image()

if __name__ == '__main__':
    root = tk.Tk()
    torch.cuda.empty_cache()
    CHECKPOINT_PATH = filedialog.askopenfilename()
    try:
        print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))
    except TypeError:
        print("input not given, closing")
        exit()
    # DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    DEVICE = 'cpu'
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)    
    app = App(root)
    root.mainloop()