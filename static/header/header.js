$(document).ready(function(){

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



