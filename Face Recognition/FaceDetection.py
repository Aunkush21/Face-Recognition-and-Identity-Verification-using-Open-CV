import cv2
import face_recognition

known_face_encodings = []
face_names = []

person1_image = face_recognition.load_image_file("D:\\Aunkush\\Python\\Face Detection\\Face Detection 1\\Aunkush.jpeg")
person2_image = face_recognition.load_image_file("D:\\Aunkush\\Python\\Face Detection\\Face Detection 1\\SRK.jpg")

person1_encoding = face_recognition.face_encodings(person1_image)[0]
person2_encoding = face_recognition.face_encodings(person2_image)[0]

known_face_encodings.append(person1_encoding)
known_face_encodings.append(person2_encoding)

face_names.append("Aunkush Barua")
face_names.append("Sharukh Khan")


capture = cv2.VideoCapture(0)

while True:

    ret, frame = capture.read()

    face_locations = face_recognition.face_locations(frame)
    current_face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, current_face_encodings):
    
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = face_names[match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("FaceDetection", frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

capture.release()
cv2.destroyAllWindows()
