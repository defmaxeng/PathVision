from nuscenes.nuscenes import NuScenes
import os
import cv2

nusc = NuScenes(version='v1.0-mini', dataroot='C:\\Users\\maxen\\OneDrive\\Documents\\GitHub\\PathVision\\v1.0-mini', verbose=True)
sample = nusc.sample[0] 
cam_data_token = sample['data']['CAM_FRONT']
cam_data = nusc.get('sample_data', cam_data_token)
image_path = os.path.join(nusc.dataroot, cam_data['filename'])
img = cv2.imread(image_path)
cv2.imshow('Front Camera', img)
cv2.waitKey(0)
cv2.destroyAllWindows()