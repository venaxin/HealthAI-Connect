const closeChat = document.getElementById("closeChat");
const userInput = document.getElementById("userInput");
const sendMessage = document.getElementById("sendMessage");
const chatButton = document.getElementById("chatButton");
const chatBox = document.getElementById("chatBox");
const chatHeader = document.querySelector(".chatheader");
const chatLog = document.querySelector(".chatLog");

let instructionsShown = false;

chatButton.addEventListener("click", () => {
  chatButton.classList.add("hidden");

  if (!instructionsShown) {
    displayDefaultInstructions();
    instructionsShown = true;
  }

  chatBox.classList.add("show");

  // Reset styles before starting the animation sequence
  chatBox.style.opacity = "0";
  chatHeader.style.opacity = "0";
  chatLog.style.height = "0";

  setTimeout(() => {
    setTimeout(() => {
      chatBox.style.opacity = "1";
      chatBox.style.transform = "translateX(0)";
      setTimeout(() => {
        chatHeader.style.opacity = "1";
      }, 0);
      setTimeout(() => {
        chatLog.style.transition = "height 0.5s";
        chatLog.style.height = "430px";
      }, 700); // Adjust timing for chat log height increase
    }, 300); // Adjust delay before chat box animation starts
  }, 0); // Adjust delay before starting the animation sequence
});

closeChat.addEventListener("click", () => {
  chatBox.style.opacity = "0";
  chatLog.style.height = "0";
  instructionsShown = false; // Reset the instructionsShown flag

  // Show the chat button again after the animation
  setTimeout(() => {
    chatBox.classList.remove("show");
    chatButton.classList.remove("hidden");
  }, 500); // Adjust timing for chat box to slide out and chat button to reappear
});

sendMessage.addEventListener("click", sendMessageHandler);

userInput.addEventListener("keypress", (event) => {
  if (event.key === "Enter") {
    sendMessageHandler();
  }
});

function sendMessageHandler() {
  const userMessage = userInput.value;
  if (userMessage.trim() !== "") {
    appendMessage("User", userMessage);
    userInput.value = "";

    // Fetch response from the server
    fetch("/send", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: userMessage }),
    })
      .then((response) => response.json())
      .then((data) => {
        appendMessage("Chatbot", data.response);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }
}

function appendMessage(sender, message) {
  const messageElement = document.createElement("div");
  messageElement.classList.add("message");
  if (sender === "User") {
    messageElement.classList.add("user-message");
    messageElement.innerHTML = message;
  } else if (sender === "Chatbot") {
    messageElement.classList.add("chatbot-message");
    messageElement.innerHTML = message;
  }

  chatLog.appendChild(messageElement);
  chatLog.scrollTop = chatLog.scrollHeight;
}

let isFirstDefaultMessage = true; // Variable to track the first default message

function displayDefaultInstructions() {
  if (isFirstDefaultMessage) {
    // Add default instructions to the chat log only if it's the first message
    instructionsShown = true;
    const defaultInstructions =
      "Welcome! I am Medibot, Ask me anything about health or medical concerns. I'm here to help!";
    const defaultMessageElement = document.createElement("div");
    defaultMessageElement.classList.add("message");
    defaultMessageElement.classList.add("chatbot-message"); // Adding chatbot message class
    defaultMessageElement.innerHTML = defaultInstructions;

    defaultMessageElement.style.marginTop = "40px";
    isFirstDefaultMessage = false; // Set the flag to false after adding the margin

    chatLog.appendChild(defaultMessageElement);
    chatLog.scrollTop = chatLog.scrollHeight;
  }
}
