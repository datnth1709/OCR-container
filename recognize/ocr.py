__author__ = Dat Nguyen-Thanh'
__copyright__ = "Copyright 2020"

import os
import torch.utils.data
from recognize.utils import *
from recognize.dataset import RawDataset, AlignCollate
from recognize.model import Model
from recognize.detection import get_detector, get_textbox
from recognize.recognition import get_text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Reader(object):
    def __init__(self,FeatureExtraction='ResNet', PAD=False, Prediction='CTC',
                 SequenceModeling='BiLSTM', Transformation='TPS', batch_max_length=25,
                 batch_size=64, character='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                 hidden_size=256, image_folder='data\\OCR-data-labelled\\OCR-data\\validate\\val',
                 imgH=32, imgW=100, input_channel=1, num_class=37, num_fiducial=20, num_gpu=1,
                 output_channel=512, rgb=False, saved_model='recognize/models/recognize_model/TPS-ResNet-BiLSTM-CTC/best_accuracy.pth',
                 sensitive=False, workers=4):
        if 'CTC' in Prediction:
            self.converter = CTCLabelConverter(character)
        else:
            self.converter = AttnLabelConverter(character)
        self.num_class = len(self.converter.character)
        self.model = Model()
        self.model = torch.nn.DataParallel(self.model).to(device)

        #load model
        self.model.load_state_dict(torch.load(saved_model, map_location=device))

        #load detector
        detector_path = os.path.join('recognize/models/detect_model', 'craft_mlt_25k.pth')
        self.detector = get_detector(detector_path, device)

        self.imgH = imgH
        self.imgW = imgW
        self.batch_max_length = batch_max_length
        self.batch_size = batch_size
        self.workers = workers
        self.PAD = PAD
        self.image_folder = image_folder
        self.character = character

    def detect(self, img, min_size=20, text_threshold=0.7, low_text=0.4, \
               link_threshold=0.4, canvas_size=2560, mag_ratio=1., \
               slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, \
               width_ths=0.5, add_margin=0.1, reformat=True):
        if reformat:
            img, img_cv_grey = reformat_input(img)

        text_box = get_textbox(self.detector, img, canvas_size, mag_ratio,\
                               text_threshold, link_threshold, low_text,\
                               False, device)
        horizontal_list, free_list = group_text_box(text_box, slope_ths,\
                                                    ycenter_ths, height_ths,\
                                                    width_ths, add_margin)

        if min_size:
            horizontal_list = [i for i in horizontal_list if max(i[1]-i[0],i[3]-i[2]) > min_size]
            free_list = [i for i in free_list if max(diff([c[0] for c in i]), diff([c[1] for c in i]))>min_size]

        return horizontal_list, free_list

    def recognize(self, img_cv_grey, horizontal_list=None, free_list=None,\
                  decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                  workers = 0, allowlist = None, blocklist = None, detail = 1,\
                  paragraph = False, contrast_ths = 0.1,adjust_contrast = 0.5, \
                  filter_ths = 0.003, reformat=True):
        if reformat:
            img, img_cv_grey = reformat_input(img_cv_grey)

        if (horizontal_list==None) and (free_list==None):
            y_max, x_max = img_cv_grey.shape
            ratio = x_max/y_max
            max_width = int(self.imgH*ratio)
            crop_img = cv2.resize(img_cv_grey, (max_width, self.imgH), interpolation =  Image.ANTIALIAS)
            image_list = [([[0,0],[x_max,0],[x_max,y_max],[0,y_max]] ,crop_img)]
        else:
            image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey, model_height = self.imgH)

        result = get_text(character=self.character, imgH=self.imgH, imgW=int(max_width), recognizer=self.model, converter=self.converter,\
                          image_list=image_list, ignore_char='', decoder=decoder, beamWidth=beamWidth, batch_size=batch_size, \
                          contrast_ths=contrast_ths, adjust_contrast=adjust_contrast, filter_ths=filter_ths, workers=1, device=device)
        if paragraph:
            result = get_paragraph(result)
        if detail == 0:
            return [item[1] for item in result]
        else:
            return result

    def recognize_from_batch(self): #batch of many images
        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(imgH=self.imgH, imgW=self.imgW, keep_ratio_with_pad=self.PAD)
        demo_data = RawDataset(root=self.image_folder, imgW=self.imgW, imgH=self.imgH)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=self.batch_size,
            shuffle=False,
            num_workers=int(self.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)
        print("demo_loader: ", demo_loader)
        self.model.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([self.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, self.batch_max_length + 1).fill_(0).to(device)

                #predict
                preds = self.model(image, text_for_pred)
                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = self.converter.decode(preds_index, preds_size)
                print("preds_str: ", preds_str)

    def readtext(self, image, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                 workers = 0, allowlist = None, blocklist = None, detail = 1,\
                 paragraph = False, min_size = 20,\
                 contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                 text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                 canvas_size = 2560, mag_ratio = 1.,\
                 slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                 width_ths = 0.5, add_margin = 0.1):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        '''
        img, img_cv_grey = reformat_input(image)

        horizontal_list, free_list = self.detect(img, min_size, text_threshold,\
                                                 low_text, link_threshold,\
                                                 canvas_size, mag_ratio,\
                                                 slope_ths, ycenter_ths,\
                                                 height_ths,width_ths,\
                                                 add_margin, False)

        result = self.recognize(img_cv_grey, horizontal_list, free_list,\
                                decoder, beamWidth, batch_size,\
                                workers, allowlist, blocklist, detail,\
                                paragraph, contrast_ths, adjust_contrast,\
                                filter_ths, False)

        return result