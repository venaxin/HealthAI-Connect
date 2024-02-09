document.addEventListener('DOMContentLoaded', function() {
    var ul = document.getElementById('messages');
    var socket = io.connect('http://' + document.domain + ':' + location.port);
    scrollToBottomSmoothly(ul);
    // Use the passed username directly without user input
    var username = document.getElementById('username').value;
    document.getElementById('form').onsubmit = function() {
        var input = document.getElementById('input');
        socket.emit('message', {'username': username, 'content': input.value});
        input.value = '';
        return false;
    };

    var defaultPage = document.getElementById('defaultPage');
        var messagesList = document.getElementById('messages');

        // Function to check and update the visibility of the default page
        function updateDefaultPageVisibility() {
            if (messagesList.children.length === 0) {
                defaultPage.style.display = 'block';
            } else {
                defaultPage.style.display = 'none';
            }
        }

    // Initial check on page load
    updateDefaultPageVisibility();
    
    socket.on('message', function(data) {
    var messageContent = document.createElement('div');
    messageContent.classList.add('message-content');

    var strong = document.createElement('strong');
    strong.textContent = data.username;

    var p = document.createElement('p');
    p.classList.add('content');
    p.textContent = data.content;

    messageContent.appendChild(strong);
    messageContent.appendChild(p);

    var li = document.createElement('li');
    li.className = data.username === username ? 'right' : '';
    li.appendChild(messageContent);

    ul.appendChild(li);

    updateDefaultPageVisibility();

    setTimeout(function() {
        scrollToBottomSmoothly(ul);
    }, 100);
});
});
function scrollToBottomSmoothly(element) {
    var duration = 500; // Scroll duration in milliseconds
    var start = element.scrollTop;
    var end = element.scrollHeight - element.clientHeight;
    var startTime;

    function scroll(timestamp) {
        if (!startTime) {
            startTime = timestamp;
        }

        var elapsed = timestamp - startTime;
        var progress = Math.min(elapsed / duration, 1);
        element.scrollTop = start + (end - start) * progress;

        if (progress < 1) {
            requestAnimationFrame(scroll);
        }
    }

    requestAnimationFrame(scroll);
}
document.getElementById('clearChatsBtn').addEventListener('click', function(event) {
    var confirmDelete = confirm('Are you sure you want to clear all chats?');
    if (!confirmDelete) {
        event.preventDefault();
    }
});
function goBack() {
    window.history.back();
}
