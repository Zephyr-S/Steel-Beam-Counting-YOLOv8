from PIL import Image

image = Image.open('/home/zihaolim/Desktop/openvino_env/yolov8-0918/fake_image_0823_2.jpg')

image.show()
image.save("/home/zihaolim/Desktop/result.jpg")


