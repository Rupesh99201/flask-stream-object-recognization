$(document).ready(function(){
  var canvas = document.querySelector("#canvasElement");
  var context = canvas.getContext('2d');
  const video = document.querySelector("#videoElement");

  video.width = 400;
  video.height = 300; 

  var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

  if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
      .then(function (stream) {
          video.srcObject = stream;
          video.play();
      })
      .catch(function (err0r) {
          console.log('Error while catching frame'+err0r);
      });
  }

  const FPS = 7;
  setInterval(() => {
      width=video.width;
      height=video.height;
      context.drawImage(video, 0, 0, width , height );
      var data = canvas.toDataURL('image/jpeg', 0.5);
      context.clearRect(0, 0, width,height );
      socket.emit('catch-frame', data);
  }, 1000/FPS);

  socket.on('response_back', function(image){
          photo.setAttribute('src', image );
          
  });
});