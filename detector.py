from keras_retinanet import models
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.utils.image import preprocess_image, resize_image
import numpy as np


class Detector:
	SCORE_CUTTOFF = 0.5

	def __init__(self, weight_path="resnet50_coco_best_v2.1.0.h5"):
		label_list = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

		self.label_list = label_list

		# backbone_name = "resnet50"
		# backbone = models.backbone(backbone_name)

		# num_classes = len(label_list)
		# num_anchors = None # models.retinanet L270
		# modifier = None # create_models

		# self.model = backbone.retinanet(num_classes, num_anchors=num_anchors, modifier=modifier)

		# self.model.load_weights(weight_file, by_name=True, skip_mismatch=True)

		# anchor_params = None
		# self.prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)


		self.prediction_model = models.load_model(weight_path, backbone_name='resnet50')

	def predict(self, image):
		"""Pass in an image get bounding boxes

			Args:
				image: a numpy array containing the image, as returned by 'cv2.imread'
					* BGR
					* rows, columns and channels (channels last)
					* uint8

			Returns:
				boxes, scores, names

				See the following for an example of usage
				https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb
		"""

		image = preprocess_image(image)
		image, scale = resize_image(image)

		all_boxes, all_scores, all_labels = self.prediction_model.predict_on_batch(np.expand_dims(image, axis=0))

		all_boxes /= scale
		#names = [self.names[int(label)] for label in labels]

		# remove batch dimension
		all_boxes = all_boxes.squeeze(axis=0)
		all_scores = all_scores.squeeze(axis=0)
		all_labels = all_labels.squeeze(axis=0)

		# filter to good results
		boxes = []
		scores = []
		labels = []

		for box, score, label in zip(all_boxes, all_scores, all_labels):

			# stop at cuttoff (results are sorted by score)
			if score < 0.5:
				break

			boxes.append(box.astype(int))
			scores.append(score)
			labels.append(self.label_list[label])


		return boxes, scores, labels