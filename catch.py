import cv2
frame_count=0
# 選擇第二隻攝影機
cap = cv2.VideoCapture(0)
count = 1
while(True):
  # 從攝影機擷取一張影像
  ret, frame = cap.read()

  if ret ==True:
  # 顯示圖片
    	cv2.imshow('frame', frame)

    	if frame_count%60==0:
           cv2.imwrite('/home/user/vic 小程式/CNN_classify/book_photo/BW_faces/img'+str(count)+'.jpg',frame)
           count+=1
  


  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  frame_count+=1
# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()