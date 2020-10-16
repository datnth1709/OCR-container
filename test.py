import os
import cv2
from recognize import ocr
reader = ocr.Reader()

test_images = "your images path"
for image in os.listdir(test_images):
    result1 = reader.readtext(path+file_name)
    frame = cv2.imread(path + file_name)
    for box in result1:
        cordinates = box[0]
        cv2.putText(frame, box[1], (int(cordinates[0][0]), int(cordinates[0][1])+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        for i, point in enumerate(cordinates):
            frame = cv2.line(frame, (int(cordinates[i-1][0]), int(cordinates[i-1][1])),(int(point[0]), int(point[1])), color, thickness)
    cv2.imshow("frame", frame)
    print(80*"*")
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
        break