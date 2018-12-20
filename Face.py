import cv2

face_cas = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img =cv2.imread("images.jpg")
grey_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face = face_cas.detectMultiScale(grey_image,scaleFactor = 1.05,minNeighbors = 5)


for a,b,c,d in face:
    cv2.rectangle(img,(a,b),(a+c,b+d),(0,220,0),3)


print(face)



cv2.imshow("GREY",img)
cv2.waitKey()
cv2.destroyAllWindows()
