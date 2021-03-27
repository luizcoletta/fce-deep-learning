import time

from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.segnet import vgg_segnet
from keras_segmentation.models.unet import mobilenet_unet
from keras_segmentation.models.pspnet import vgg_pspnet

start_time = time.time()

              # Model 1     Model 2       Model 3          model 4
models_list = ['VGG-Unet', 'VGG-Segnet', 'Mobilenet-Unet', 'VGG-Pspnet']
selected_model = 3
num_classes = 4
img_height = 224
img_width = 224
num_epochs = 50
#mobilenet so aceita tamanhos em 224

model = []
chkp_dir_list = ["saved_models/vgg_unet_1", "saved_models/vgg_segnet", "saved_models/mobilenet_unet"]
print(">> Initializing and training " + models_list[selected_model-1] + " model" )
if models_list[selected_model-1] == 'VGG-Unet':
    model = vgg_unet(n_classes=num_classes, input_height=img_height, input_width=img_width)
else:
    if models_list[selected_model-1] == 'VGG-Segnet':
        model = vgg_segnet(n_classes=num_classes, input_height=img_height, input_width=img_width)
    else:
        if models_list[selected_model-1] == 'Mobilenet-Unet':
            model = mobilenet_unet(n_classes=num_classes, input_height=img_height, input_width=img_width)
        else:
            if models_list[selected_model-1] == 'VGG-Pspnet':
                model = vgg_pspnet(n_classes=num_classes, input_height=img_height, input_width=img_width)

model.train(
    train_images="data/dataset_weeds/images_prepped_train/",
    train_annotations="data/dataset_weeds/annotations_prepped_train/",
    checkpoints_path=chkp_dir_list[selected_model-1], epochs=num_epochs
)

print("--- %s seconds ---" % (time.time() - start_time))