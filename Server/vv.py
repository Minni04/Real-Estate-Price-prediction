import cv2
import face_recognition

# Load an image with faces you want to recognize
image_of_person_1 = face_recognition.load_image_file("person_1.jpg")
person_1_face_encoding = face_recognition.face_encodings(image_of_person_1)[0]

image_of_person_2 = face_recognition.load_image_file("person_2.jpg")
person_2_face_encoding = face_recognition.face_encodings(image_of_person_2)[0]

# Create an array of known face encodings and corresponding labels
known_face_encodings = [
    person_1_face_encoding,
    person_2_face_encoding
]

known_face_labels = [
    "Person 1",
    "Person 2"
]

# Open the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame from the webcam
    _, frame = video_capture.read()

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in the current frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match is found, use the label of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_labels[first_match_index]

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()