function addTask() {
  const taskInput = document.getElementById("taskInput");
  const taskText = taskInput.value.trim();

  if (taskText !== "") {
    const taskList = document.getElementById("taskList");
    const li = document.createElement("li");
    li.className = "todo-item";
    li.innerHTML = `<div class="taskListHeader"><input type="checkbox" onchange="markDone(this)">
                <span class="task">${taskText}</span></div>
                <button class="delete-btn" onclick="removeTask(this)"><i class="fas fa-trash-alt"></i></button>`;
    taskList.appendChild(li);
    taskInput.value = "";
  }
}

function markDone(checkbox) {
  const taskText = checkbox.nextElementSibling;
  if (checkbox.checked) {
    taskText.style.textDecoration = "line-through";
  } else {
    taskText.style.textDecoration = "none";
  }
}

function removeTask(btn) {
  const li = btn.parentNode;
  li.parentNode.removeChild(li);
}