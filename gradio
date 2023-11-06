import gradio as gr
from ultralytics import YOLO
import cv2
# from PIL import Image
from predict import predict
# from config import Config
# cfg = Config()

# agesss = {'less18':'teenager','over60':'old people','18-60':'18-60'}
def snap(image):

    model = YOLO(r'/home/dingzf2/projects/gender_age/resources/yolov8x.pt')
    model_gender = YOLO(r'/home/dingzf2/projects/gender_age/resources/gender_mix1.pt')
    model_age = YOLO(r'/home/dingzf2/projects/gender_age/resources/best_ages0.pt')
    res = model(image)
    image_orig = res[0].orig_img
    img_hight = image_orig.shape[0]
    img_weight = image_orig.shape[1]
    for box in res[0].boxes:
        t = box.data.tolist()[0]
        label = res[0].names[int(t[5])]
        if (label == 'person'):
            t0  = int(t[0]-0.02*img_weight) if (int(t[0]-0.02*img_weight))>0 else 0
            t2  = int(t[2]+0.02*img_weight) if int(t[2]+0.02*img_weight)<img_weight else img_weight
            t1  = int(t[1]-0.02*img_hight)  if int(t[1]-0.02*img_hight)>0 else 0
            t3  = int(t[3]+0.02*img_hight) if int(t[3]+0.02*img_hight)<img_hight else img_hight
            crop = image_orig[t1:t3, t0:t2]
            result = model_gender(crop,imgsz=224)
            result2 = model_age(crop,imgsz=224)
            ressss = result[0].probs.top1
            ressss2 = result2[0].probs.top1
            top1conf = result[0].probs.top1conf.cpu().numpy()
            label2 = result[0].names[ressss]
            label23= result2[0].names[ressss2]
            
            cv2.rectangle(image_orig, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (255,0,0), 3) 
            cv2.putText(image_orig, label2+' '+str(label23), (int(t[0]) + 10, int(t[1])+20), 0, 1, [255,0,0], thickness=2,
                            lineType=cv2.LINE_AA)
    return image_orig

def snap2(video):
    predict(video)
    
    return 'output.mp4'





if __name__ == "__main__":
    demo_video = gr.Interface(
    snap,
    inputs=["image"], outputs=["image"],
    
    # examples=[
    #     ['fight0.mp4'],
    #     ["fall1.mp4"],
    # ]
)
    demo_image = gr.Interface(
    snap2,
    [gr.Video(input='video')],
    [gr.Video(output='video')])
    demo = gr.TabbedInterface([demo_video,demo_image],["性别年龄_视频","性别年龄_图像"])
    demo.launch(share=True,server_name='10.10.30.30',server_port=7862)
