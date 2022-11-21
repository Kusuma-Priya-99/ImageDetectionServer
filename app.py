from flask import Flask, render_template, request
from PIL import Image
import os
import io
import base64
import matplotlib.pyplot as plt
import torch
import cv2
import yaml
from torchvision import transforms
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open('data/hyp.scratch.mask.yaml') as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)
weigths = torch.load('tools/yolov7-mask.pt')
model = weigths['model']
model = model.half().to(device)
_ = model.eval()

PEOPLE_FOLDER = os.path.join('yolov7', 'people_photo')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

def image(model,image) :

    image = letterbox(image, 640, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    #image = image.type(torch.bfloat16)
    model = model.type(torch.float32)
    output = model(image)

    inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']

    bases = torch.cat([bases, sem_output], dim=1)
    nb, _, height, width = image.shape
    names = model.names
    pooler_scale = model.pooler_scale
    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)

    output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)
    #%%

    output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)
    #%%
    pred, pred_masks = output[0], output_mask[0]
    base = bases[0]
    bboxes = Boxes(pred[:, :4])
    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
    pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
    pred_masks_np = pred_masks.detach().cpu().numpy()
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)
    pnimg = nimg.copy()

    for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
        if conf < 0.25:
            continue
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

        pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # label = '%s %.3f' % (names[int(cls)], conf)
        # t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        # c2 = bbox[0] + t_size[0], bbox[1] - t_size[1] - 3
        # pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), c2, color, -1, cv2.LINE_AA)  # filled
        # pnimg = cv2.putText(pnimg, label, (bbox[0], bbox[1] - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    return pnimg

def serve_pil_image(pil_img):

    img_io = io.BytesIO()
    pil_img.save(img_io, 'jpeg', quality=100)
    img_io.seek(0)
    img = base64.b64encode(img_io.getvalue()).decode('ascii')
    img_tag = f'<img src="data:image/jpg;base64,{img}" class="img-fluid"/>'
    return img_tag

@app.route("/", methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    global model
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        npimg = np.fromstring(file1.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        #img = Image.fromarray(img.astype("uint8"))
        print(file1)
        display_image = image(model, img)
        img_tag = serve_pil_image(Image.fromarray(display_image))
        return render_template("index.html", image=img_tag)

    return render_template("index.html")

app.run()


