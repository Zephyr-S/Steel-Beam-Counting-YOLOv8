import os
from pathlib import Path

#---------------------------------parameters-------------------------------------#
import argparse

# models_dir = Path('/home/zihaolim/Desktop/openvino_env/yolov8-0918/models-0918')

parser = argparse.ArgumentParser(description='InputPath - OutputPath - ModelPath - Confidence')
#parser.add_argument('--inputdir', type=str, default=input_path, help='The path to input images folder.')
#parser.add_argument('--outputdir', type=str, default=output_pth, help='The path to output images folder.')
parser.add_argument('--inputdir', type=str, default='./input', help='The path to input images folder.')
parser.add_argument('--outputdir', type=str, default='./output', help='The path to output images folder.')
parser.add_argument('--modeldir', type=str, default=Path(r'.\models'), help='The path to models folder.')
parser.add_argument('--conf_threshold', type=float, default=0.7, help='The lowest confidence of result.')
args = parser.parse_args()

output_pth = Path(args.outputdir)
if not os.path.exists(output_pth):
    os.makedirs(output_pth)
    
print(args)

#---------------------------------parameters-------------------------------------#


from ultralytics import YOLO
from utility import *

#---------------------------------bbx-------------------------------------#
'''DET_MODEL_NAME = "det-0822"
det_model = YOLO(models_dir / f'{DET_MODEL_NAME}.pt')
# det_model = YOLO('/content/drive/MyDrive/Projects/yolov8-bbx-0822/runs/detect/train/weights/best.pt')
label_map = det_model.model.names

# res = det_model(IMAGE_PATH)
# Image.fromarray(res[0].plot()[:, :, ::-1])

det_model_path = models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
if not det_model_path.exists():
    det_model.export(format="openvino", dynamic=True, half=False)'''
#---------------------------------bbx-------------------------------------#

SEG_MODEL_NAME = "seg-0822"
# SEG_MODEL_NAME = "best"
#SEG_MODEL_NAME = "best-model-0925"

seg_model = YOLO(args.modeldir / f'{SEG_MODEL_NAME}.pt')
label_map = seg_model.model.names
# seg_model = YOLO('/content/drive/MyDrive/Projects/yolov8-training/0822/runs/segment/train2/weights/best.pt')
# res = seg_model(IMAGE_PATH)
# Image.fromarray(res[0].plot()[:, :, ::-1])



seg_model_path = args.modeldir / f"{SEG_MODEL_NAME}_openvino_model/{SEG_MODEL_NAME}.xml"
if not seg_model_path.exists():
    seg_model.export(format="openvino", dynamic=True, half=False)


import ipywidgets as widgets
from openvino.runtime import Core

core = Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)



from postprocessing import *
from openvino.runtime import Core, Model
import cv2 as cv

core = Core()

#---------------------------------bbx-------------------------------------#
"""det_ov_model = core.read_model(det_model_path)
if device.value != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
det_compiled_model = core.compile_model(det_ov_model, device.value)




input_image = np.array(Image.open(IMAGE_PATH))
detections = detect(input_image, det_compiled_model)[0]
# image_with_boxes, count_sum = draw_results_xz(detections, input_image, label_map)  # if need postprocessing
# image_with_boxes = draw_results(detections, input_image, label_map)
image_with_boxes, count_result = draw_count_results(detections, input_image, label_map)
result_img = Image.fromarray(image_with_boxes)
img_name = 'T%d_O%d_I%d.jpg' % (count_sum['2'], count_sum['1'], count_sum['0'])
result_img.save(os.path.join(output_pth, img_name))
print(img_name, "is saved in", output_pth)"""
#---------------------------------bbx-------------------------------------#


seg_ov_model = core.read_model(seg_model_path)
if device.value != "CPU":
    seg_ov_model.reshape({0: [1, 3, 640, 640]})
seg_compiled_model = core.compile_model(seg_ov_model, device.value)

for root, dirs, files in os.walk(args.inputdir):
	for file_name in files:
		print(file_name)
		if file_name[-3:] == 'png':
			tmp_file_name = file_name[:-3]+'jpg'
			img = Image.open(os.path.join(args.inputdir, file_name))
			img.convert('RGB').save((os.path.join(args.inputdir, tmp_file_name)))
			file_name = tmp_file_name
		input_image = np.array(Image.open(os.path.join(args.inputdir, file_name)))  # input img
		input_image = cv.resize(input_image, (640, 640))

		detections = detect( args.conf_threshold, input_image, seg_compiled_model)[0]  # start detect
		# image_with_masks, count_sum = draw_results_xz(detections, input_image, label_map)  # if need postprocessing
		# image_with_masks = draw_results(detections, input_image, label_map)
		image_with_masks, count_result = draw_count_results(detections, input_image, label_map)
		result_img = Image.fromarray(image_with_masks)
		img_name = 'T%d_O%d_I%d.jpg' % (count_result['2'], count_result['1'], count_result['0'])  # T O I
		result_img.save(os.path.join(args.outputdir, img_name))
		print(img_name, "is saved in", args.outputdir)

