const totalAppointmentsElement = document.getElementsByClassName("req-apts")[0];
const totalAppointments = currentAppointmentsElement.innerText;
const currentAppointmentsElement =
  document.getElementsByClassName("accept-apts")[0];
const currentAppointments = currentAppointmentsElement.innerText;
console.log(currentAppointments);
const fillPercentage = (currentAppointments / maxAppointments) * 100;
const gauge = document.querySelector(".gauge");
gauge.style.setProperty("--fill", `${fillPercentage}%`);

const gaugeMeter = document.querySelector(".gauge-meter");
gaugeMeter.textContent = currentAppointments;
const aptLeft = document.getElementById("apt-left");
aptLeft.textContent = `Appointments Left: ${currentAppointments}`;

