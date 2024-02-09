$(document).ready(function(){

    (function () {

        const link = document.querySelectorAll('nav > .hover-this');
        const cursor = document.querySelector('.cursor');
    
        const animateit = function (e) {
              const span = this.querySelector('span');
              const { offsetX: x, offsetY: y } = e,
              { offsetWidth: width, offsetHeight: height } = this,
    
              move = 25,
              xMove = x / width * (move * 2) - move,
              yMove = y / height * (move * 2) - move;
    
              span.style.transform = `translate(${xMove}px, ${yMove}px)`;
    
              if (e.type === 'mouseleave') span.style.transform = '';
        };
    
        const editCursor = e => {
              const { clientX: x, clientY: y } = e;
              cursor.style.left = x + 'px';
              cursor.style.top = y + 'px';
        };
    
        link.forEach(b => b.addEventListener('mousemove', animateit));
        link.forEach(b => b.addEventListener('mouseleave', animateit));
        window.addEventListener('mousemove', editCursor);
    
    })();



    var showHeaderAt = 130;

    var win = $(window),
            body = $('body');

    // Show the fixed header only on larger screen devices

    if(win.width() > 400){

        // When we scroll more than 150px down, we set the
        // "fixed" class on the body element.

        win.on('scroll', function(e){

            if(win.scrollTop() > showHeaderAt) {
                body.addClass('fixed');
            }
            else {
                body.removeClass('fixed');
            }
        });

    }

});
const changingImage = document.getElementById('changingImage');

const imageSources = [
  '../static/images/HealthAI-l.gif'
];

let currentIndex = 0; 
function changeImage() {
  changingImage.setAttribute('src', imageSources[currentIndex]);

  currentIndex = (currentIndex + 1) % imageSources.length;

  setTimeout(changeImage, 3000); 
}

setTimeout(changeImage, 1000);



