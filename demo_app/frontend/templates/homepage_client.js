document.addEventListener("DOMContentLoaded", function () {
  // First Message.
  const first_message = document.createElement("li");
  first_message.innerHTML =
    "Hi, I'm Travel Bot and I can help you with booking!";

  // Add class for styling.
  first_message.setAttribute("class", "bot-message");

  setTimeout(function () {
    // Bot Message bubble.
    document.querySelector("ul").append(first_message);
    return false;
  }, 1000);

  // Chat Box.
  document.querySelector("form").onsubmit = function () {
    // Checking for blank submission.
    let blank_submit = document.querySelector("input").value;

    if (blank_submit == "" || blank_submit == null) {
      //Blank submit check.
      alert("Blank Submission was made");
    } else if (/^\s*$/.test(blank_submit)) {
      // Blank Lines Check.
      alert("Blank Submission was made");
    } else {
      // Handle User input.
      let user_input = document.querySelector("input").value;
      console.log(user_input);

      // Send POST to server.
      const data = { user_input: user_input };

      fetch("/chat", {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify(data),
      })
        .then((response) => response.json())

        .then((data) => {
          console.log("Success:", data);

          // Bot Response.
          let bot_response = document.createElement("li");
          var bot_message = data["bot_response"];

          bot_response.innerHTML = bot_message;
          bot_response.setAttribute("class", "bot-message");

          setTimeout(function () {
            // Bot Message bubble.
            document.querySelector("ul").append(bot_response);
            return false;
          }, 1000);
        })
        .catch((error) => {
          console.error("Error:", error);
        });

      // User Message bubble.
      const user_response = document.createElement("li");
      user_response.innerHTML = user_input;

      // Add class for styling.
      user_response.setAttribute("class", "user-message");
      document.querySelector("ul").append(user_response);

      // Empty form.
      document.querySelector("input").value = "";

      return false;
    }
  };
});
