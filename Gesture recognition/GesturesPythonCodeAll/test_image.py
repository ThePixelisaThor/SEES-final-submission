import cv2

img = cv2.imread("Untitled.png", cv2.IMREAD_GRAYSCALE)  # Read in grayscale
print(img)
cv2.imshow("1", img)
img = cv2.resize(img, (5, 2))

print(img)
cv2.imshow("2", img)
img = img.reshape(10)
print(img)
cv2.imshow("3", img)

cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows() 