
const aboutSection = document.querySelector('.about-section');
const benContainer = document.querySelector('.benefits-container');
const whySection = document.querySelector('.why-section');

const aboutObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      aboutSection.classList.add('slide-in');
    }
  });
});

const benObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      benContainer.classList.add('slide-in');
    }
  });
});

const whyObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      whySection.classList.add('appear');
    }
  });
});

aboutObserver.observe(aboutSection);
benObserver.observe(benContainer);
whyObserver.observe(whySection);

function isInViewport(element) {
  const rect = element.getBoundingClientRect();
  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
  );
}

function handleTechItems() {
  const techItems = document.querySelectorAll('.tech-item');

  function handleScrollTech() {
    techItems.forEach(item => {
      if (isInViewport(item)) {
        item.classList.add('appear');
      }
    });
  }

  window.addEventListener('scroll', handleScrollTech);
  window.addEventListener('load', handleScrollTech);
}
handleTechItems();
