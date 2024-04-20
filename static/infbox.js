function showContent(contentId) {
    // Hide all content divs
    var contents = document.querySelectorAll('.content');
    for (var i = 0; i < contents.length; i++) {
      contents[i].style.display = 'none';
    }
    // Show the selected content
    document.getElementById(contentId + '-content').style.display = 'block';
  }

  // Show overview content by default
  showContent('overview');

  document.querySelectorAll('.info-box-fixed .info-box-limiter nav a').forEach(anchor => {
    anchor.addEventListener('click', function() {
      // Remove 'clicked' class from all anchors
      document.querySelectorAll('.info-box-fixed .info-box-limiter nav a').forEach(a => {
        a.classList.remove('clicked');
      });
      // Add 'clicked' class to the clicked anchor
      this.classList.add('clicked');
    });
  });
  