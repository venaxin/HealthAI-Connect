const screenWidth = window.innerWidth;
const boxWidth = 100; // Set the box width

const boxesInRow = Math.floor(screenWidth / boxWidth);

document.documentElement.style.setProperty("--boxes-in-row", boxesInRow);

const boxesInColumn = Math.ceil(window.innerHeight / 100);

for (let i = 0; i < boxesInRow * boxesInColumn; i++) {
  const box = document.createElement("div");
  box.className = "box";
  document.body.appendChild(box);

  let isHovered = false;
  let isBlack = true; // Initially set the color as black

  // Function to toggle box color on mouseout
  const toggleBoxColor = () => {
    if (isBlack) {
      box.style.backgroundColor = "#090243";
    } else {
      box.style.backgroundColor = "#000";
    }
  };

  // Add event listeners to toggle box color on mouseover
  box.addEventListener("mouseover", () => {
    isHovered = true;
    box.style.backgroundColor = "cyan";
  });

  box.addEventListener("mouseout", () => {
    isHovered = false;
    isBlack = !isBlack; // Toggle between black and purple colors
    setTimeout(() => {
      if (!isHovered) {
        toggleBoxColor();
      }
    }, 100); // Delay in milliseconds (100 milliseconds in this case)
  });

  // Initially set the color of the box
  toggleBoxColor();
}

function validatePassword() {
  const password = document.getElementById("password").value;
  const regex = /^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;

  if (!regex.test(password)) {
    alert(
      "Password must contain at least 8 characters, including 1 uppercase letter, 1 number, and 1 special symbol."
    );
    return false; // Prevent form submission
  }
  return true; // Allow form submission
}

function togglePasswordVisibility() {
  const passwordField = document.getElementById("password");
  const toggleButton = document.getElementById("togglePassword");

  if (passwordField.type === "password") {
    passwordField.type = "text";
    toggleButton.innerHTML =
      '<i class="fa fa-eye-slash" aria-hidden="true"></i>';
  } else {
    passwordField.type = "password";
    toggleButton.innerHTML = '<i class="fa fa-eye" aria-hidden="true"></i>';
  }
}

//password visibility filter
const root = document.documentElement;
const eye = document.getElementById("eyeball");
const beam = document.getElementById("beam");
const passwordInput = document.getElementById("password");

root.addEventListener("mousemove", (e) => {
  let rect = beam.getBoundingClientRect();
  let mouseX = rect.right + rect.width / 2;
  let mouseY = rect.top + rect.height / 2;
  let rad = Math.atan2(mouseX - e.pageX, mouseY - e.pageY);
  let degrees = rad * (20 / Math.PI) * -1 - 350;

  root.style.setProperty("--beamDegrees", `${degrees}deg`);
});

eye.addEventListener("click", (e) => {
  e.preventDefault();
  document.body.classList.toggle("show-password");
  passwordInput.type = passwordInput.type === "password" ? "text" : "password";
  passwordInput.focus();
});
