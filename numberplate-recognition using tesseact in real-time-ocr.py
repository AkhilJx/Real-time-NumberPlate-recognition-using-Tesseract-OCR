import cv2
import pytesseract
import csv
import re
import datetime
import os

# Path to Tesseract executable (change this if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Path to Haarcascade XML file for license plate detection (change this if necessary)
cascade_path = 'haarcascade_russian_plate_number.xml'

# Initialize the cascade classifier
cascade = cv2.CascadeClassifier(cascade_path)

def is_valid_text(text):
    # pattern = r'^[a-zA-Z0-9\s]+$'
    pattern = r'^[A-Za-z]{2}\s?\d{1,2}\s?[A-Za-z]{1,2}\s?\d{1,4}$'
    return bool(re.match(pattern, text))

# Function to read the number plate from an image
def read_number_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in plates:
        plate_img = gray[y:y + h, x:x + w]
        plate_text = pytesseract.image_to_string(plate_img, config='--psm 7')
        return plate_text.strip()

# Initialize the video capture
video_capture = cv2.VideoCapture(0)  # Change the argument to the video file path if reading from a file

# Specify the CSV file name
csv_file_name = 'number_plates.csv'

# Check if the CSV file exists
csv_exists = os.path.isfile(csv_file_name)

# Open the CSV file in append mode
csv_file = open(csv_file_name, 'a', newline='')
csv_writer = csv.writer(csv_file)

# If the CSV file doesn't exist, write the header
if not csv_exists:
    csv_writer.writerow(['Vehicle Number', 'Timestamp'])

# Initialize the set to store unique values
unique_plates = set()

# Process frames from the video
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect number plates
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected number plate
    for (x, y, w, h) in plates:
        plate_img = frame[y:y + h, x:x + w]
        plate_text = pytesseract.image_to_string(plate_img, config='--psm 7')
        plate_text = plate_text.strip()

        # Draw a bounding box around the number plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if is_valid_text(plate_text) and plate_text not in unique_plates:
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')

            # Write the text and timestamp to the CSV file
            csv_writer.writerow([plate_text, timestamp])
            unique_plates.add(plate_text)
            print(f'{plate_text} - {timestamp}')

    # Display the resulting frame
    cv2.imshow('Number Plate Reader', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the CSV file
video_capture.release()
csv_file.close()

# Destroy OpenCV windows
cv2.destroyAllWindows()
