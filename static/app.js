const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messages = document.querySelector("#messages");
const sendButton = form.querySelector(".send-button");
const charCount = document.querySelector("#char-count");
const clearButton = document.querySelector("#clear-chat");

const welcomeMarkup = `
  <div class="message-row bot-row">
    <div class="message-avatar" aria-hidden="true">n.</div>
    <div class="message-content">
      <span class="message-name">NLPBot <span>JUST NOW</span></span>
      <div class="bubble bot-bubble">Hey there! I’m NLPBot 👋<br>Pick a prompt below or ask me anything about AI, NLP, or Python.</div>
    </div>
  </div>`;

function scrollToLatest() {
  messages.scrollTop = messages.scrollHeight;
}

function addMessage(text, sender) {
  const row = document.createElement("div");
  row.className = `message-row ${sender}-row`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = sender === "bot" ? "n." : "you";

  const content = document.createElement("div");
  content.className = "message-content";

  const name = document.createElement("span");
  name.className = "message-name";
  name.textContent = sender === "bot" ? "NLPBot" : "YOU";

  const bubble = document.createElement("div");
  bubble.className = `bubble ${sender}-bubble`;
  bubble.textContent = text;

  content.append(name, bubble);
  row.append(avatar, content);
  messages.append(row);
  scrollToLatest();
  return row;
}

function showTyping() {
  const row = addMessage("", "bot");
  row.querySelector(".message-name").textContent = "NLPBot is thinking";
  const bubble = row.querySelector(".bubble");
  bubble.setAttribute("aria-label", "NLPBot is typing");
  bubble.innerHTML = '<span class="typing-dots" aria-hidden="true"><i></i><i></i><i></i></span>';
  return row;
}

async function sendMessage(value) {
  const message = value.trim();
  if (!message || sendButton.disabled) return;

  addMessage(message, "user");
  input.value = "";
  input.style.height = "auto";
  charCount.textContent = "0/1000";
  sendButton.disabled = true;
  input.disabled = true;
  const typingRow = showTyping();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const result = await response.json();
    typingRow.remove();
    if (!response.ok) throw new Error(result.error || "The message could not be sent.");
    addMessage(result.response, "bot");
  } catch (error) {
    typingRow.remove();
    addMessage(error.message || "I couldn't reach NLPBot. Please try again.", "bot");
  } finally {
    sendButton.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("input", () => {
  charCount.textContent = `${input.value.length}/1000`;
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 90)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.textContent));
});

clearButton.addEventListener("click", () => {
  messages.innerHTML = welcomeMarkup;
  input.focus();
});
