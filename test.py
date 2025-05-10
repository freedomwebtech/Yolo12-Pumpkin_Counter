# Import required libraries
import cv2
from ultralytics import YOLO
import cvzone


model = YOLO('')
names=model.names


# Open video file or webcam
cap = cv2.VideoCapture("vid5.mp4")  # Use 0 for webcam

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")
        
# Create a named OpenCV window and set the mouse callback
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)
frame_count=0 
while True:
    # Read video frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 3 != 0:
        continue 
    frame = cv2.resize(frame, (1020,600))
    

    # Detect and track persons (class 0)
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        for track_id,box,class_id in zip(ids,boxes,class_ids):
            x1,y1,x2,y2=box
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2
            name=names[class_id]
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, name, (x1 + 3, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            


            
        

       
    
    # Show the frame
    cv2.imshow("RGB", frame)
    
    # Press ESC to exit
    if cv2.waitKey(0) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
