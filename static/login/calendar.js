document.addEventListener("DOMContentLoaded", function () {
  const calendarDiv = document.getElementById("calendar");
  const currentDate = new Date();
  let currentMonth = currentDate.getMonth();
  let currentYear = currentDate.getFullYear();

  function renderCalendar(month, year) {
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const monthNames = [
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December",
    ];

    let calendarHTML = "";
    calendarHTML += `<div class="calendar-buttons">
                    <button id="prevMonthBtn">&larr;</button>
                    <h2>${monthNames[month]} ${year}</h2>
                    <button id="nextMonthBtn">&rarr;</button>
                 </div>`;
    calendarHTML += `<table><tr><th>Sun</th><th>Mon</th><th>Tue</th><th>Wed</th><th>Thu</th><th>Fri</th><th>Sat</th></tr><tr>`;

    let dayCount = 0;
    for (
      let i = 1;
      i <= daysInMonth + (new Date(year, month, 1).getDay() % 7);
      i++
    ) {
      if (i > new Date(year, month, 1).getDay() && dayCount < daysInMonth) {
        dayCount++;
        if (
          dayCount === currentDate.getDate() &&
          month === currentDate.getMonth() &&
          year === currentDate.getFullYear()
        ) {
          calendarHTML += `<td class="today">${dayCount}</td>`;
        } else {
          calendarHTML += `<td>${dayCount}</td>`;
        }
      } else {
        calendarHTML += `<td></td>`;
      }

      if (
        i % 7 === 0 &&
        i < daysInMonth + (new Date(year, month, 1).getDay() % 7)
      ) {
        calendarHTML += `</tr><tr>`;
      }
    }
    calendarHTML += `</tr></table>`;

    calendarDiv.innerHTML = calendarHTML;

    document
      .getElementById("prevMonthBtn")
      .addEventListener("click", function () {
        if (currentMonth === 0) {
          currentMonth = 11;
          currentYear -= 1;
        } else {
          currentMonth -= 1;
        }
        renderCalendar(currentMonth, currentYear);
      });

    document
      .getElementById("nextMonthBtn")
      .addEventListener("click", function () {
        if (currentMonth === 11) {
          currentMonth = 0;
          currentYear += 1;
        } else {
          currentMonth += 1;
        }
        renderCalendar(currentMonth, currentYear);
      });
  }

  renderCalendar(currentMonth, currentYear);

  // Add this part to ensure the current date is shaded initially
  const today = calendarDiv.querySelector('.today');
  if (today) {
    today.style.backgroundColor = 'rgb(94 171 255)'; // Adjust the color to your preference
  }
});
