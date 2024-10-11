
DATASETS = [ "cassava","cifar10","cifar100","colorectal_histology","eurosat","food101","oxford_flowers102","oxford_iiit_pet","resisc45","stanford_dogs","sun397","svhn_cropped"]

TASKS_A = ["colorectal_histology","cifar10","eurosat","oxford_iiit_pet","oxford_flowers102", "sun397"]
TASKS_B = ["resisc45","cifar100","cassava","svhn_cropped","stanford_dogs","food101"]

VITB16_INIT = '/home/ekansh/repos/share/vision-models/experiments/ViTB16/hugging_face/cassava/sgd/steps_10000_500/cosine_1e-3/seed_0/init'
VITB16 = {
  "cassava": '/home/ekansh/repos/share/vision-models/experiments/ViTB16/hugging_face/cassava/sgd/steps_10000_500/cosine_1e-3/seed_0', 
  "cifar10": '/home/ekansh/repos/share/vision-models/experiments/ViTB16/hugging_face/cifar10/sgd/steps_10000_500/cosine_1e-3/seed_0', 
  "cifar100": '/home/ekansh/repos/share/vision-models/experiments/ViTB16/hugging_face/cifar100/sgd/steps_10000_500/cosine_1e-3/seed_0', 
  "colorectal_histology": '/home/ekansh/repos/share/vision-models/experiments/ViTB16/hugging_face/colorectal_histology/sgd/steps_10000_500/cosine_1e-3/seed_0', 
  "eurosat": '/home/ekansh/repos/share/vision-models/experiments/ViTB16/hugging_face/eurosat/sgd/steps_10000_500/cosine_1e-3/seed_0', 
  "food101": '/home/ekansh/repos/share/vision-models/experiments/ViTB16/hugging_face/food101/sgd/steps_10000_500/cosine_1e-3/seed_0', 
  "oxford_flowers102": '/home/ekansh/repos/share/vision-models/experiments/ViTB16/hugging_face/oxford_flowers102/sgd/steps_10000_500/cosine_1e-3/seed_0', 
  "oxford_iiit_pet": '/home/ekansh/repos/share/vision-models/experiments/ViTB16/hugging_face/oxford_iiit_pet/sgd/steps_10000_500/cosine_1e-3/seed_0', 
  "resisc45": '/home/ekansh/repos/share/vision-models/experiments/ViTB16/hugging_face/resisc45/sgd/steps_10000_500/cosine_1e-3/seed_0', 
  "stanford_dogs": '/home/ekansh/repos/share/vision-models/experiments/ViTB16/hugging_face/stanford_dogs/sgd/steps_10000_500/cosine_1e-3/seed_0', 
  "sun397": '/home/ekansh/repos/share/vision-models/experiments/ViTB16/hugging_face/sun397/sgd/steps_10000_500/cosine_1e-3/seed_0', 
  "svhn_cropped": '/home/ekansh/repos/share/vision-models/experiments/ViTB16/hugging_face/svhn_cropped/sgd/steps_10000_500/cosine_1e-3/seed_0', 
}

VITMAEB16_INIT = '/home/ekansh/repos/share/vision-models/experiments/ViTmaeB16/hugging_face/cassava/sgd/steps_10000_500/cosine_1e-2/seed_0/init'
VITMAEB16 = {
  "cassava": '/home/ekansh/repos/share/vision-models/experiments/ViTmaeB16/hugging_face/cassava/sgd/steps_10000_500/cosine_1e-2/seed_0', 
  "cifar10": '/home/ekansh/repos/share/vision-models/experiments/ViTmaeB16/hugging_face/cifar10/sgd/steps_10000_500/cosine_1e-2/seed_0', 
  "cifar100": '/home/ekansh/repos/share/vision-models/experiments/ViTmaeB16/hugging_face/cifar100/sgd/steps_10000_500/cosine_1e-2/seed_0', 
  "colorectal_histology": '/home/ekansh/repos/share/vision-models/experiments/ViTmaeB16/hugging_face/colorectal_histology/sgd/steps_10000_500/cosine_1e-2/seed_0', 
  "eurosat": '/home/ekansh/repos/share/vision-models/experiments/ViTmaeB16/hugging_face/eurosat/sgd/steps_10000_500/cosine_1e-2/seed_0', 
  "food101": '/home/ekansh/repos/share/vision-models/experiments/ViTmaeB16/hugging_face/food101/sgd/steps_10000_500/cosine_1e-2/seed_0', 
  "oxford_flowers102": '/home/ekansh/repos/share/vision-models/experiments/ViTmaeB16/hugging_face/oxford_flowers102/sgd/steps_10000_500/cosine_1e-2/seed_0', 
  "oxford_iiit_pet": '/home/ekansh/repos/share/vision-models/experiments/ViTmaeB16/hugging_face/oxford_iiit_pet/sgd/steps_10000_500/cosine_1e-2/seed_0', 
  "resisc45": '/home/ekansh/repos/share/vision-models/experiments/ViTmaeB16/hugging_face/resisc45/sgd/steps_10000_500/cosine_1e-2/seed_0', 
  "stanford_dogs": '/home/ekansh/repos/share/vision-models/experiments/ViTmaeB16/hugging_face/stanford_dogs/sgd/steps_10000_500/cosine_1e-2/seed_0', 
  "sun397": '/home/ekansh/repos/share/vision-models/experiments/ViTmaeB16/hugging_face/sun397/sgd/steps_10000_500/cosine_1e-2/seed_0', 
  "svhn_cropped": '/home/ekansh/repos/share/vision-models/experiments/ViTmaeB16/hugging_face/svhn_cropped/sgd/steps_10000_500/cosine_1e-2/seed_0', 
}

VGG16X2_NONLOCAL_TASK_A_INIT = "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_13/cassava/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/init"
VGG16X2_NONLOCAL_TASK_B_INIT = "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_15/cassava/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/init"

VGG16X2_A = {
  "cassava": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_13/cassava/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "cifar10": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_13/cifar10/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "cifar100": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_13/cifar100/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "colorectal_histology": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_13/colorectal_histology/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "eurosat": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_13/eurosat/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "food101": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_13/food101/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "oxford_flowers102": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_13/oxford_flowers102/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "oxford_iiit_pet": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_13/oxford_iiit_pet/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "resisc45": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_13/resisc45/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "stanford_dogs": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_13/stanford_dogs/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "sun397": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_13/sun397/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "svhn_cropped": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_13/svhn_cropped/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
}

VGG16X2_B = {
  "cassava": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_15/cassava/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "cifar10": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_15/cifar10/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "cifar100": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_15/cifar100/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "colorectal_histology": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_15/colorectal_histology/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "eurosat": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_15/eurosat/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "food101": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_15/food101/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "oxford_flowers102": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_15/oxford_flowers102/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "oxford_iiit_pet": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_15/oxford_iiit_pet/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "resisc45": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_15/resisc45/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "stanford_dogs": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_15/stanford_dogs/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "sun397": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_15/sun397/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
  "svhn_cropped": "/home/ekansh/repos/share/vision-models/experiments/VGG16x2/ILSVRC_15/svhn_cropped/sgd/trained_classifier/steps_8000_500/cosine_1e-2/seed_0/", 
}


