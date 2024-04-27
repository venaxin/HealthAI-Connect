const dropdownToggles = document.querySelectorAll(".dropdown-toggle");

dropdownToggles.forEach((toggle) => {
toggle.addEventListener("click", function (event) {
    event.preventDefault(); // Prevent default link behavior
    const dropdown = this.nextElementSibling;
    if (dropdown && dropdown.classList.contains("dropdown")) {
    dropdown.classList.toggle("active");
    }
});
});

document.querySelectorAll(".smooth-scroll").forEach((anchor) => {
anchor.addEventListener("click", function (e) {
    e.preventDefault();

    const targetId = this.getAttribute("href").substring(1);
    const target = document.getElementById(targetId);

    if (target) {
    window.scrollTo({
        top: target.offsetTop,
        behavior: "smooth",
    });
    }
});
});