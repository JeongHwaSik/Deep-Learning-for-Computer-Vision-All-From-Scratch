# A5. Object Detection

## A5-1. One-Stage Object Detector
I implemented [FCOS:Fully-Convolutional One-Stage Object Detector](https://arxiv.org/pdf/1904.01355) from scratch as a one-stage object detection model and trained it on the PASCAL VOC 2007 dataset. 
This dataset is relatively small, containing 20 categories with annotated bounding boxes.
Unlike classification tasks, mean Average Precision (mAP) is used as the validation metric for evaluation.

<img width="1200" alt="Screenshot 2024-12-25 at 11 20 11 PM" src="https://github.com/user-attachments/assets/76d34d83-e2ee-4285-9902-7d9905d68044" />

(Detection sucks... Debug here ☹️)

<img width="1200" alt="Screenshot 2024-12-25 at 11 27 13 PM" src="https://github.com/user-attachments/assets/d4b4854e-5b11-479b-ad56-4d9c11963ffc" />

See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A5/one_stage_detector.ipynb) for more details about FCOS implementation!

## A5-2. Two-Stage Object Detector
I implemented a two-stage object detector based on [Faster R-CNN](https://arxiv.org/pdf/1506.01497), which comprises two main modules: the Region Proposal Network (RPN) and Fast R-CNN.
I used FCOS as a backbone instead of Fast R-CNN .
As with previous section in 5-1, I used the PASCAL VOC 2007 dataset and evaluated performance using mean Average Precision (mAP) as the metric. 
 
(Detection sucks... Debug here ☹️)

<img width="1200" alt="Screenshot 2024-12-25 at 11 31 39 PM" src="https://github.com/user-attachments/assets/bf73bd33-8372-48d3-984d-bd6718b5065c" />

See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A5/two_stage_detector.ipynb) for more details about Faster R-CNN with FCOS implementation!


<br>
</br>
