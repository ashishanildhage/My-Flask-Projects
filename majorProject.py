import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# # Initialize mediapipe pose class.
# mp_pose = mp.solutions.pose

# # Setup the Pose function for images - independently for the images standalone processing.
# pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# # Setup the Pose function for videos - for video processing.
# pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,
#                           min_tracking_confidence=0.7)

# # Initialize mediapipe drawing class - to draw the landmarks points.
# mp_drawing = mp.solutions.drawing_utils

# def detectPose(image_pose, pose, draw=False, display=False):
    
#     original_image = image_pose.copy()
    
#     image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    
#     resultant = pose.process(image_in_RGB)

#     if resultant.pose_landmarks and draw:    

#         mp_drawing.draw_landmarks(image=original_image, landmark_list=resultant.pose_landmarks,
#                                   connections=mp_pose.POSE_CONNECTIONS,
#                                   landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
#                                                                                thickness=3, circle_radius=3),
#                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
#                                                                                thickness=2, circle_radius=2))

#     if display:
        
#         plt.figure(figsize=[22,22])
#         plt.subplot(121);plt.imshow(image_pose[:,:,::-1]);plt.title("Input Image");plt.axis('off');
#         plt.subplot(122);plt.imshow(original_image[:,:,::-1]);plt.title("Pose detected Image");plt.axis('off');

#     else:
#         return original_image, results

# # Here we will read our image from the specified path to detect the pose
# image_path = 'media/sample2.jpg'
# output = cv2.imread(image_path)
# detectPose(output, pose_image, draw=True, display=True)


mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawingStyles
mp_pose=mp.solutions.pose

cap=cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False) as pose_video:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read video")
            continue
        image.flags.writeable = False
        image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results=pose.process(image)
        
        image.flags.writeable = True
        image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmark_drawing_style())

        cv2.imshow("Pose Estimation", cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
