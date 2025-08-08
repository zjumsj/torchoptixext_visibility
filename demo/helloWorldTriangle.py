import cv2
import torchoptixext_visibility

img = torchoptixext_visibility.run_sample_code() # -> uint8 tensor HxWx4
img = img.cpu().detach().numpy()
# Coordinate conversion: OpenGL uses bottom-left origin while most image coordinates use top-left origin.
img = img[::-1]

bgra_image = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
cv2.imwrite('triangle.png', bgra_image)
print('Dump triangle.png')
