# Bird's eye view

How to build this?

<img src="./bev.jpg" alt="Tesla Bird's Eye View" width="150">

**Logitech HD Pro C920**  
Diagonal FOV: 78°  
Resolution: 1920x1080  
Aspect Ratio: 16:9  

**Jetson Orin**   
GPU: 32 tensor cores, 1020Hhz  
CPU: 6-core Arm 64-bit CPU 1.5Mb L2 + 4MB L3, 1.7Ghz  
Memory: 8GB 128-bit LPDDR5 102 GB/s  
Power: 25W  

<!-- Starts the camera -->
curl -H "Content-Type: application/json" -d '{"method":"startRecMode","params":[],"id":1,"version":"1.0"}' http://192.168.122.1:8080/sony/camera

<!-- Get available api -->
curl -H "Content-Type: application/json" -d '{"method":"getAvailableApiList","params":[],"id":1,"version":"1.0"}' http://192.168.122.1:8080/sony/camera

**API Methods on startRecMode**
- `getVersions`
- `getMethodTypes`
- `getApplicationInfo`
- `getAvailableApiList`
- `getEvent`
- `actTakePicture`
- `stopRecMode`
- `startLiveview`
- `stopLiveview`
- `actZoom`
- `setSelfTimer`
- `getSelfTimer`
- `getAvailableSelfTimer`
- `getSupportedSelfTimer`
- `getExposureCompensation`
- `getAvailableExposureCompensation`
- `getSupportedExposureCompensation`
- `setShootMode`
- `getShootMode`
- `getAvailableShootMode`
- `getSupportedShootMode`
- `getSupportedFlashMode`

<!-- Starts live view -->
curl -H "Content-Type: application/json" -d '{"method":"startLiveview","params":[],"id":1,"version":"1.0"}' http://192.168.122.1:8080/sony/camera

curl -H "Content-Type: application/json" -d '{"method":"stopLiveview","params":[],"id":1,"version":"1.0"}' http://192.168.122.1:8080/sony/camera
