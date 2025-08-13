import cv2
import numpy as np
import pytesseract
from sklearn.ensemble import RandomForestClassifier
import face_recognition
import pytesseract
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import face_recognition


# Load the font classification model
classifier = RandomForestClassifier()
classifier.load('font_classifier.model')

# Load the face recognition model
known_face_encodings = []
known_face_names = []
for i in range(10):
    image = face_recognition.load_image_file("person"+str(i+1)+".jpg")
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append("Person "+str(i+1))

# Open the video capture device
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    # Find contours of the number plate
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Extract the number plate from the frame
    number_plate = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) == 4:
            number_plate = approx
            break

    # Apply perspective transform to get a bird's eye view of the number plate
    if number_plate is not None:
        plate_pts = number_plate.reshape(4, 2)
        plate_rect = np.zeros((4, 2), dtype=np.float32)
        plate_rect[:2] = plate_pts[:2]
        plate_rect[2] = plate_pts[2]
        plate_rect[3] = plate_pts[3]
        width = 360
        height = 80
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(plate_rect, dst)
        plate_image = cv2.warpPerspective(frame, M, (width, height))

        # Apply OCR to recognize the characters on the number plate
        text = pytesseract.image_to_string(plate_image, config='--psm 11')

        # Predict font type
        features = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = gray[y:y+h, x:x+w]
            resized_roi = cv2.resize(roi, (50, 50))
            features.append(np.array(resized_roi).flatten())
        if len(features) > 0:
            font_type = classifier.predict(features)[0]
        else:
            font_type = 'Unknown'

        # Detect faces in the image
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        face_names = []

        # Recognize faces in the image
        for face_encoding in face_encodings:
    # Compare the face encoding with known face encodings to see if it matches
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
    
    # If there's a match, use the known name for the face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    face_names.append(name)

    # Predict font type
    if umber_plate is not None:
         x, y, w, h = cv2.boundingRect(number_plate)      
    plate_img = gray[y:y+h, x:x+w]
    features = extract_features(plate_img)
    font_type = ''
         
   
    if len(features) == 0:
        font_type = 'Unknown'
    else:
        font_type = classifier.predict(features)[0]

    # Display the results
    print('Number Plate: ', text)
    print('Font Type: ', font_type)
    print('Faces Detected: ', face_names)

    



# Load the image
img = cv2.imread("image.jpg")

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to preprocess the image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Perform OCR to get the text from the image
text = pytesseract.image_to_string(thresh)

# Define a regular expression pattern to match phone numbers
pattern = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")

# Search for phone numbers in the OCR output
matches = pattern.findall(text)

# Check if any of the phone numbers are in the format of a talking phone (e.g. "Hey Siri")
talking_phone = False
for match in matches:
    if "hey" in match.lower() or "siri" in match.lower() or "ok" in match.lower() or "google" in match.lower():
        talking_phone = True
        break

# If a talking phone was detected, print a warning message
if talking_phone:
    print("WARNING: Driver may be using a talking phone while driving.")
else:
    print("No talking phone detected.")



# Load the font classification model
classifier = RandomForestClassifier()
classifier.load('font_classifier.model')

# Load the face recognition model
known_face_encodings = []
known_face_names = []
for i in range(10):
    image = face_recognition.load_image_file("person"+str(i+1)+".jpg")
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append("Person "+str(i+1))

# Open the video capture device
cap = cv2.VideoCapture(0)

# Initialize variables for triple riding detection
TRIPLE_RIDING_AREA = [(500, 300), (700, 300), (700, 500), (500, 500)]
TRIPLE_RIDING_COLOR = (0, 0, 255)
triple_riding_detected = False

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    # Find contours of the number plate
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Extract the number plate from the frame
    number_plate = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) == 4:
            number_plate = approx
            break

    # Apply perspective transform to get a bird's eye view of the number plate
    if number_plate is not None:
        plate_pts = number_plate.reshape(4, 2)
        plate_rect = np.zeros((4, 2), dtype=np.float32)
        plate_rect[:2] = plate_pts[:2]
        plate_rect[2] = plate_pts[2]
        plate_rect[3] = plate_pts[3]
        width = 360
        height = 80
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(plate_rect, dst)
        plate_image = cv2.warpPerspective(frame, M, (width, height))

        # Apply OCR to recognize the characters on the number plate
        text = pytesseract.image_to_string(plate_image, config='--psm 11')

        # Predict font type
        features = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = gray[y:y+h, x:x+w]
            resized_roi = cv2.resize(roi, (50, 50))
            features.append(np.array(resized_roi).flatten())
        if len(features) > 0:
            font_type = classifier.predict(features)[0]
        else:
            font_type = 'Unknown'

        # Detect faces in the image
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

# Check for triple riding violation
    if len(face_locations) > 2:
        print('Triple Riding Detected!')
    # Draw bounding boxes around faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    else:
    # Recognize faces in the image
        face_names = []
    for face_encoding in face_encodings:
        # Compare the face encoding with known face encodings to see if it matches
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # If there's a match, use the known name for the face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
        
        # Draw bounding box around face
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
        # Display the results
    print('Faces Detected: ', face_names)

# Show the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()