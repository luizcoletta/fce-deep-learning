import time

from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.segnet import vgg_segnet
from keras_segmentation.models.unet import mobilenet_unet
from keras_segmentation.models.pspnet import vgg_pspnet

def init_model(sel_model, num_classes):
    model = []
    print(">> Initializing " + sel_model[0] + " model")
    if sel_model[0] == 'vgg_unet':
        model = vgg_unet(n_classes=num_classes, input_height=sel_model[1], input_width=sel_model[2])
    else:
        if sel_model[0] == 'vgg_segnet':
            model = vgg_segnet(n_classes=num_classes, input_height=sel_model[1], input_width=sel_model[2])
        else:
            if sel_model[0] == 'mobilenet_unet':
                model = mobilenet_unet(n_classes=num_classes, input_height=sel_model[1], input_width=sel_model[2])
            else:
                if sel_model[0] == 'vgg_pspnet':
                    model = vgg_pspnet(n_classes=num_classes, input_height=sel_model[1], input_width=sel_model[2])
    return model

def train_model(model, num_epochs, img_path, ann_path, chkp_dir):
    print(">> Training the selected model")
    start_time = time.time()
    model.train(
        train_images=img_path,
        train_annotations=ann_path,
        checkpoints_path=chkp_dir, epochs=num_epochs
    )
    print(">> Model trained after " + str(time.time() - start_time) + " seconds")
    return model

selected_model = 3
models_list = [("vgg_unet",         320, 640), # 0
               ("vgg_segnet",       320, 640), # 1
               ("mobilenet_unet",   224, 224), # 2
               ("vgg_pspnet",       384, 768)] # 3

num_classes = 4
num_epochs = 50

img_path = "data/dataset_weeds/images_prepped_train/"
ann_path = "data/dataset_weeds/annotations_prepped_train/"
chkp_dir_list = "saved_models/" + models_list[selected_model][0] + "_1"

model = init_model(models_list[selected_model], num_classes)
train_model(model, num_epochs, img_path, ann_path, chkp_dir_list)