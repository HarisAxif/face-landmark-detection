import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
my_drawing_specs = mp_drawing.DrawingSpec(color = (0, 255, 255), thickness = 1)

cap = cv2.VideoCapture("test1.mp4")  # capture video from webcam

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as face_mesh:

    while cap.isOpened():
        success, image = cap.read() # read the images from camera and saved in image
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_styles
                .get_default_face_mesh_tesselation_style() # to get default style by mediapipe
            )

            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = None,
                connection_drawing_spec = my_drawing_specs
                # .get_default_face_mesh_contours_style() # to get default style by mediapipe
            ) 


        cv2.imshow("Video Capture", cv2.flip(image, 1))

        if cv2.waitKey(100) & 0xFF == ord("q"): # press q for 100 ms to quit video
            break

cap.release()
cv2.destroyAllWindows()
