import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened(): 
    print("Unable to read camera feed")
backSub = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction
    fgMask = backSub.apply(frame)

    # Threshold to remove noise
    _, thresh = cv2.threshold(fgMask, 25, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over contours and draw rectangles around moving objects
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust this value to filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display output
    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground Mask', fgMask)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
