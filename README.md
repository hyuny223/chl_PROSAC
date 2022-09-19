완벽하게 구현하지는 못 하여 estimation에 편차가 조금 있다.  
\+ opencv보다 성능이 좋지 않다. 어느 부분이 문제일까?   
# Result
[ left image ]
![image](https://user-images.githubusercontent.com/58837749/190558050-0e47aaf8-03c4-412a-a672-1830c347daf6.png)
[ right image ]
![image](https://user-images.githubusercontent.com/58837749/190558079-bf8685bd-efa9-4e51-a000-bffd14f6dae4.png)
[ my homography with PROSAC ]
![image](https://user-images.githubusercontent.com/58837749/191072124-606c9403-6d65-4017-ac1a-845e01530396.png)
[ opencv homography ]
![image](https://user-images.githubusercontent.com/58837749/190558120-8d200c40-70f7-4807-b79b-2faf878559b0.png)

# Profiler
![image](https://user-images.githubusercontent.com/58837749/191080346-8e4cb140-dcb9-4b0a-a9f8-9d576e7dcc52.png)
전체적인 프로파일러를 살펴보자면, RANSAC을 수행하는데 꽤나 많은 시간이 걸리는 것을 알 수 있다. 대략 2/100초 정도 걸린다. 어디서 이정도의 시간이 걸릴까?  
![image](https://user-images.githubusercontent.com/58837749/191081412-fb179fb6-f674-4597-b515-356070c859ea.png)
범인은 computeSVD였다. OpenCV는 SVD 한번 돌리는 시간에 이미 답을 찾는다. Inliers 기준이 아니라, RMSE 기준으로 돌려봐도 OpenCV와는 1/10 차이가 난다. 
